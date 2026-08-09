"""
speech_gate.py
--------------
An energy-based speech/non-speech mask over the waveform, computed once and
handed to both components.

Why this exists
---------------
Both detectors were trained exclusively on speech utterances — ASVspoof 2019 LA
for Component 1, PartialSpoof v1.2 for Component 2. Neither ever saw silence,
room tone, breath or music as an input it had to classify. Their output on such
frames is not a low-confidence guess; it is out-of-distribution and arbitrary.
Measured on this checkpoint pair:

    input                     C1 P(bonafide)    C2 mean P(spoof) per frame
    digital silence, 6 s          0.317              0.880
    white noise, 6 s              0.737              0.938

Both flag silence as spoof, and Component 2 does so at a confidence higher than
most genuine splices. That is the mechanism behind real recordings being
reported as fake: a normal recording opens and closes with room tone and pauses
between phrases, every one of those regions clears the frame threshold, several
run past the minimum-duration floor, and the file comes back "Partially
Tampered" — or, once the pauses dominate, "Fully AI-Generated".

Restricting both components to speech frames is therefore not a heuristic patch
on top of the models; it is confining them to the domain they were measured on.

Why energy and not a neural VAD
-------------------------------
A learned VAD would add a third model, a third download and a third failure
mode to a system whose whole design point is that its two models are
independent and traceable. The decision this gate has to make — "is there
speech-level signal here at all" — is one a relative energy threshold makes
reliably, and being wrong in the conservative direction (keeping a quiet
speech frame) costs nothing, because the frame then simply goes to the
detectors as before.

The threshold is relative to the file's own loud percentile rather than
absolute, so a quietly recorded interview and a loud studio file are gated the
same way. The absolute floor is a second gate that catches the degenerate case
of a file that is entirely near-silent, where a purely relative threshold would
declare its loudest hiss to be speech.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
DEFAULT_HOP_S = 0.02      # matches Component 2's frame rate
DEFAULT_WIN_S = 0.025     # matches WavLM's receptive field per frame

DEFAULTS = {
    "enabled": True,
    # Frames this many dB below the file's loud percentile are non-speech.
    "rel_db": -35.0,
    # ...and frames below this absolute level are non-speech regardless, so a
    # file containing nothing but hiss does not have its hiss promoted.
    "abs_dbfs": -55.0,
    # The "loud" reference. 95th rather than max, so one click does not raise
    # the reference for the whole file.
    "ref_percentile": 95.0,
    # Hysteresis. A raw 20 ms energy decision flickers: speech dips below any
    # fixed level at every stop consonant and inter-word pause, which would
    # chop one spoof region into a dozen sub-second fragments and make the
    # reported time ranges unreadable. Non-speech gaps shorter than
    # `min_gap_s` are bridged, and every speech run is then extended by
    # `hangover_s` at each end so that quiet onsets and decays stay inside it.
    # This is the standard hangover/hold arrangement of an energy VAD.
    "min_gap_s": 0.20,
    "hangover_s": 0.10,
}


def _bridge_short_gaps(flags: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill runs of False shorter than `max_gap` frames that sit between two
    True runs. Leading and trailing silence is left alone - only interior
    gaps are speech that dipped, and only interior gaps risk fragmenting a
    region that should be reported as one."""
    if max_gap <= 0 or not flags.any():
        return flags
    out = flags.copy()
    true_idx = np.flatnonzero(flags)
    first, last = true_idx[0], true_idx[-1]
    i = first
    while i <= last:
        if out[i]:
            i += 1
            continue
        j = i
        while j <= last and not out[j]:
            j += 1
        if (j - i) < max_gap:
            out[i:j] = True
        i = j
    return out


def _extend(flags: np.ndarray, pad: int) -> np.ndarray:
    """Grow every True run by `pad` frames at each end (binary dilation)."""
    if pad <= 0 or not flags.any():
        return flags
    out = flags.copy()
    for shift in range(1, pad + 1):
        out[shift:] |= flags[:-shift]
        out[:-shift] |= flags[shift:]
    return out


class SpeechMask:
    """Per-frame speech/non-speech decision over one waveform.

    Frames are on a fixed 20 ms grid from the start of the file, which is the
    same grid Component 2 reports on, so the two line up without resampling.
    """

    def __init__(self, flags: np.ndarray, hop_s: float, duration_s: float,
                 params: dict):
        self.flags = flags.astype(bool)
        self.hop_s = float(hop_s)
        self.duration_s = float(duration_s)
        self.params = params

    # -- construction ---------------------------------------------------
    @classmethod
    def from_waveform(cls, wav: np.ndarray, sr: int = SAMPLE_RATE,
                      params: dict | None = None,
                      hop_s: float = DEFAULT_HOP_S,
                      win_s: float = DEFAULT_WIN_S) -> "SpeechMask":
        p = dict(DEFAULTS)
        p.update({k: v for k, v in (params or {}).items() if k in DEFAULTS})

        wav = np.asarray(wav, dtype=np.float32)
        duration_s = len(wav) / sr
        hop = max(1, int(round(hop_s * sr)))
        win = max(hop, int(round(win_s * sr)))
        n_frames = max(1, int(np.ceil(len(wav) / hop)))

        if not p["enabled"]:
            return cls(np.ones(n_frames, dtype=bool), hop_s, duration_s, p)

        # RMS in dBFS per frame. Centre each window on its frame so a frame at
        # the very start or end is not penalised by the missing half-window.
        db = np.empty(n_frames, dtype=np.float64)
        half = win // 2
        for i in range(n_frames):
            centre = i * hop + hop // 2
            lo = max(0, centre - half)
            hi = min(len(wav), centre + half)
            block = wav[lo:hi]
            rms = float(np.sqrt(np.mean(block * block))) if block.size else 0.0
            db[i] = 20.0 * np.log10(rms + 1e-10)

        ref = float(np.percentile(db, p["ref_percentile"]))
        flags = (db >= ref + p["rel_db"]) & (db >= p["abs_dbfs"])

        flags = _bridge_short_gaps(flags, int(round(p["min_gap_s"] / hop_s)))
        flags = _extend(flags, int(round(p["hangover_s"] / hop_s)))
        return cls(flags, hop_s, duration_s, p)

    # -- queries --------------------------------------------------------
    def at(self, times_s: np.ndarray) -> np.ndarray:
        """Speech flag for each of `times_s`, by nearest frame on this grid.

        Component 2's frames are anchored to their chunk's sample offset, so
        their times are on the same 20 ms grid but their count differs from
        this mask's. Looking up by time rather than by index is what keeps the
        two aligned on a long, chunked file.
        """
        idx = np.rint(np.asarray(times_s, dtype=float) / self.hop_s).astype(int)
        idx = np.clip(idx, 0, len(self.flags) - 1)
        return self.flags[idx]

    def fraction(self, start_s: float, end_s: float) -> float:
        """Fraction of the interval [start_s, end_s) that is speech."""
        lo = int(np.floor(start_s / self.hop_s))
        hi = int(np.ceil(end_s / self.hop_s))
        lo = max(0, min(lo, len(self.flags)))
        hi = max(lo + 1, min(hi, len(self.flags)))
        window = self.flags[lo:hi]
        return float(window.mean()) if window.size else 0.0

    @property
    def speech_duration_s(self) -> float:
        return float(self.flags.sum()) * self.hop_s

    @property
    def speech_ratio(self) -> float:
        return float(self.flags.mean()) if self.flags.size else 0.0

    def summary(self) -> dict:
        return {
            "enabled": bool(self.params["enabled"]),
            "speech_duration_s": round(self.speech_duration_s, 3),
            "speech_ratio": round(self.speech_ratio, 4),
            "params": self.params,
        }
