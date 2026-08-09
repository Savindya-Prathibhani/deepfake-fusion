"""
component1_aasist/infer.py
--------------------------
Inference-only wrapper around Band-Augmented AASIST (Component 1).

What was reused, verbatim
-------------------------
  * `external/aasist_model.py`   - the pinned clovaai/aasist `Model` class,
                                   copied byte-for-byte from the vendored
                                   `external/aasist/models/AASIST.py`.
  * `external/AASIST.conf`       - the same architecture hyperparameters the
                                   checkpoint was trained with.
  * The crop/pad rule below      - identical to `ASVspoofDataset._load()` at
                                   eval time (repeat-pad short, centre-crop
                                   long) and to notebook 10's `to_model_input`.
  * The scoring rule below       - identical to `evaluate.compute_loss_and_scores`
                                   on the plain-softmax branch:
                                   `softmax(logits)[:, 1]` = P(bonafide).

What was dropped, and why
-------------------------
  * `model_setup.ensure_aasist_source()` git-clones the AASIST repo at runtime.
    An app should not shell out to git on startup, so the source is vendored
    into `external/` instead. The architecture it builds is unchanged.
  * `train.build_criterion()` was only needed by notebook 10 so that
    `compute_loss_and_scores` could dispatch on the criterion type. This
    experiment's config is `loss.type: softmax`, so that dispatch always lands
    on the plain-softmax branch. Inlining those two lines removes the whole
    training module from the dependency graph without changing the number.

Long-input handling - a deliberate, documented deviation
--------------------------------------------------------
AASIST sees exactly 64600 samples (4.04 s). Evaluation centre-crops, because
ASVspoof utterances are short and the centre is representative. For an app
that accepts arbitrary audio and video, centre-cropping would silently discard
most of a 3-minute file. So long inputs are split into consecutive 64600-sample
windows, every window is scored, and the windows are aggregated (default:
mean). Files of 4.04 s or less take exactly the original single-window path,
so short-file behaviour is bit-identical to the validated pipeline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_EXTERNAL = _HERE / "external"
if str(_EXTERNAL) not in sys.path:
    sys.path.insert(0, str(_EXTERNAL))

MAX_LEN = 64600          # samples; from training.max_len in band_augment.yaml
SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Framing - same rule as ASVspoofDataset._load() with augment=False
# ---------------------------------------------------------------------------

def _fit_window(wav: np.ndarray, max_len: int = MAX_LEN) -> np.ndarray:
    """Repeat-pad if shorter than max_len, centre-crop if longer.

    Repeat-pad (not zero-pad) is load-bearing: zero-padding introduces long
    silence regions that the sinc front end reads as signal.
    """
    if len(wav) < max_len:
        return np.tile(wav, (max_len // len(wav)) + 1)[:max_len]
    if len(wav) > max_len:
        start = (len(wav) - max_len) // 2
        return wav[start:start + max_len]
    return wav


def _windows(wav: np.ndarray, max_len: int = MAX_LEN) -> list[np.ndarray]:
    """Consecutive non-overlapping analysis windows covering the whole file.

    A file at or under one window long yields exactly one window, produced by
    `_fit_window` - i.e. the original eval path, unchanged.
    """
    if len(wav) <= max_len:
        return [_fit_window(wav, max_len)]

    out: list[np.ndarray] = []
    for start in range(0, len(wav), max_len):
        chunk = wav[start:start + max_len]
        if len(chunk) < max_len:
            # Trailing remainder: repeat-pad it the same way a short file is
            # padded, rather than dropping it, so the tail of the file is
            # still examined.
            if len(chunk) < max_len // 8:
                break          # < 0.5 s of leftover: not worth a full window
            chunk = _fit_window(chunk, max_len)
        out.append(chunk)
    return out or [_fit_window(wav, max_len)]


def _window_starts(wav: np.ndarray, n_windows: int, max_len: int = MAX_LEN
                   ) -> list[int]:
    """Sample offset of each window returned by `_windows`.

    Kept next to `_windows` so the two cannot drift: a window's offset is what
    maps it onto the speech mask, and an offset that disagreed with the window
    it names would gate the wrong region.
    """
    if len(wav) <= max_len:
        return [0]
    return [i * max_len for i in range(n_windows)]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class AASISTDetector:
    """Whole-utterance real/fake detector. Independent of Component 2."""

    def __init__(self, checkpoint_path: str | Path, threshold: float,
                 device: str | None = None, aggregate: str = "mean"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = float(threshold)
        self.aggregate = aggregate

        with open(_EXTERNAL / "AASIST.conf") as f:
            conf = json.load(f)

        from aasist_model import Model as AASISTModel   # vendored, pinned

        self.model = AASISTModel(d_args=conf["model_config"]).to(self.device)

        ckpt = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False)
        # Pipeline-trained checkpoints wrap the weights; a bare state_dict is
        # also accepted so the frozen legacy `aasist_model.pth` still loads.
        state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def predict(self, wav: np.ndarray, speech_mask=None,
                min_speech_fraction: float = 0.5) -> dict:
        """wav: mono float32 @16 kHz. Returns whole-audio scores and verdict.

        `score` is P(bonafide): HIGHER means more real. A file is called fake
        when the score falls below the threshold - the same direction used
        throughout Component 1's evaluation.

        `speech_mask`: optional `speech_gate.SpeechMask`. Windows that are
        mostly non-speech are dropped before the aggregate is taken. AASIST was
        trained on ASVspoof utterances, which are speech throughout; this
        checkpoint scores six seconds of digital silence at P(bonafide) 0.317,
        i.e. confidently FAKE. Averaging that over a real recording's pauses
        drags the whole-file score below the 0.7142 threshold and reports a
        genuine file as synthetic. Dropping those windows keeps the score on
        the material the model was measured on. If no window qualifies, every
        window is used and `speech_windows_only` is False, so the fallback is
        visible in the output rather than silent.
        """
        if len(wav) == 0:
            raise ValueError("Empty waveform: nothing to analyse.")

        windows = _windows(wav)
        starts = _window_starts(wav, len(windows))

        kept = list(range(len(windows)))
        speech_windows_only = False
        if speech_mask is not None and min_speech_fraction > 0:
            qualifying = [
                i for i, s in enumerate(starts)
                if speech_mask.fraction(s / SAMPLE_RATE,
                                        (s + MAX_LEN) / SAMPLE_RATE)
                >= min_speech_fraction
            ]
            if qualifying:
                kept = qualifying
                speech_windows_only = True

        batch = torch.from_numpy(
            np.stack([windows[i] for i in kept])).float().to(self.device)

        out = self.model(batch)
        logits = out[1] if isinstance(out, tuple) else out
        probs = torch.softmax(logits.float(), dim=1)
        scores = probs[:, 1].cpu().numpy().astype(float)   # P(bonafide)

        score = float(scores.mean() if self.aggregate == "mean" else scores.min())
        is_fake = bool(score < self.threshold)

        return {
            "score_bonafide": score,
            "threshold": self.threshold,
            "verdict": "FAKE" if is_fake else "REAL",
            "is_fake": is_fake,
            "confidence": _margin_confidence(score, self.threshold),
            "n_windows": len(kept),
            "n_windows_total": len(windows),
            "speech_windows_only": speech_windows_only,
            "window_scores": [round(s, 6) for s in scores.tolist()],
            "window_aggregate": self.aggregate,
        }


def _margin_confidence(score: float, threshold: float) -> float:
    """Distance from the decision threshold, rescaled to [0, 1].

    Normalised by the distance to the far end of the score range on whichever
    side the score fell, so a score of 0.0 against a 0.71 threshold and a score
    of 1.0 against the same threshold both read as full confidence.

    This is a decision MARGIN, not a calibrated probability. Component 1's own
    XAI analysis found this model family's softmax output to be saturated, so
    treating the raw score as a probability would overstate certainty.
    """
    if score < threshold:
        denom = threshold if threshold > 0 else 1.0
        return float(min(1.0, (threshold - score) / denom))
    denom = (1.0 - threshold) if threshold < 1 else 1.0
    return float(min(1.0, (score - threshold) / denom))
