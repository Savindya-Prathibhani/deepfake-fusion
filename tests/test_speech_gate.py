"""
Unit tests for the speech gate.

The gate is pure signal processing over a numpy array — no model, no
checkpoint, no audio file — so it is testable in isolation, which matters
because it is now load-bearing for every verdict the system produces.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from speech_gate import SpeechMask, _bridge_short_gaps, _extend   # noqa: E402

SR = 16000


def tone(seconds, freq=200, amp=0.2, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds, sr=SR):
    return np.zeros(int(seconds * sr), dtype=np.float32)


# ---------------------------------------------------------------------------

def test_silence_is_not_speech():
    """The whole point: digital silence, which both detectors score as
    confidently spoof, must not reach them as evidence."""
    m = SpeechMask.from_waveform(silence(3.0))
    assert m.speech_ratio == 0.0
    assert m.speech_duration_s == 0.0


def test_speech_between_silence_is_isolated():
    wav = np.concatenate([silence(2.0), tone(3.0), silence(2.0)])
    m = SpeechMask.from_waveform(wav)
    # 3 s of tone plus 0.1 s hangover at each end.
    assert 3.0 <= m.speech_duration_s <= 3.5
    assert m.fraction(0.0, 1.5) == 0.0
    assert m.fraction(3.0, 4.0) == 1.0


def test_gate_is_relative_so_a_quiet_recording_is_not_all_silence():
    """A file recorded 30 dB down must gate the same as a loud one; an
    absolute-only threshold would discard a quiet interview entirely."""
    loud = np.concatenate([silence(1.0), tone(2.0, amp=0.5), silence(1.0)])
    quiet = np.concatenate([silence(1.0), tone(2.0, amp=0.015), silence(1.0)])
    a = SpeechMask.from_waveform(loud)
    b = SpeechMask.from_waveform(quiet)
    assert abs(a.speech_duration_s - b.speech_duration_s) < 0.25


def test_absolute_floor_stops_hiss_being_promoted_to_speech():
    """A file that is nothing but very low noise has no loud reference, so a
    purely relative threshold would call its own hiss speech."""
    rng = np.random.default_rng(0)
    hiss = (0.00005 * rng.standard_normal(SR * 3)).astype(np.float32)
    assert SpeechMask.from_waveform(hiss).speech_ratio == 0.0


def test_short_interior_gaps_are_bridged():
    """A stop consonant is a ~50 ms dip. Left alone it splits one reported
    region into two, and each half may then fall under the duration floor."""
    wav = np.concatenate([tone(1.0), silence(0.05), tone(1.0)])
    m = SpeechMask.from_waveform(wav)
    assert m.fraction(0.98, 1.07) == 1.0


def test_long_interior_gaps_are_not_bridged():
    """A genuine two-second pause is not speech and must stay excluded."""
    wav = np.concatenate([tone(1.0), silence(2.0), tone(1.0)])
    m = SpeechMask.from_waveform(wav)
    assert m.fraction(1.5, 2.5) == 0.0


def test_leading_silence_is_not_bridged():
    """Bridging is for interior dips only. Head and tail silence stays out —
    beyond the deliberate hangover, which does extend ~0.1 s into it so a
    quiet onset is not clipped."""
    wav = np.concatenate([silence(1.0), tone(2.0)])
    m = SpeechMask.from_waveform(wav)
    assert m.fraction(0.0, 0.8) == 0.0
    assert m.fraction(1.0, 2.0) == 1.0


def test_disabled_gate_passes_everything():
    """`enabled: false` must restore the pre-gate behaviour exactly, so the
    change can be isolated when comparing against earlier results."""
    m = SpeechMask.from_waveform(silence(2.0), params={"enabled": False})
    assert m.speech_ratio == 1.0


def test_lookup_by_time_is_clamped_not_wrapped():
    """Component 2's frame times can run a hop past the mask's last frame on a
    file whose length is not a whole number of frames."""
    m = SpeechMask.from_waveform(tone(1.0))
    flags = m.at(np.array([-0.5, 0.5, 999.0]))
    assert len(flags) == 3
    assert flags.dtype == bool


# ---------------------------------------------------------------------------
# The two mask post-processing primitives, directly
# ---------------------------------------------------------------------------

def test_bridge_short_gaps_leaves_edges_alone():
    f = np.array([0, 0, 1, 1, 0, 1, 1, 0, 0], dtype=bool)
    out = _bridge_short_gaps(f, max_gap=2)
    assert out.tolist() == [False, False, True, True, True, True, True, False, False]


def test_bridge_short_gaps_is_a_no_op_on_an_empty_mask():
    f = np.zeros(5, dtype=bool)
    assert _bridge_short_gaps(f, max_gap=3).tolist() == [False] * 5


def test_extend_grows_both_ends_without_running_off_the_array():
    f = np.array([0, 0, 1, 0, 0], dtype=bool)
    assert _extend(f, 1).tolist() == [False, True, True, True, False]
    assert _extend(f, 5).tolist() == [True] * 5
