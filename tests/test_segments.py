"""
Unit tests for Component 2's frames -> time-ranges conversion.

`BAMLocalizer._segments` is exercised without building the model: it only
touches numpy arrays and the two thresholds, so the instance is created with
`__new__` and the two attributes it reads are set directly. That keeps these
tests runnable without the 385 MB checkpoint or a WavLM download, while still
testing the real method rather than a copy of it.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "component2_bam"))

from component2_bam.infer import (   # noqa: E402
    FRAME_HOP_S, BAMLocalizer, _merge_close_runs,
)


def localizer(frame_threshold=0.55):
    obj = BAMLocalizer.__new__(BAMLocalizer)
    obj.frame_threshold = frame_threshold
    return obj


def frames(spec: str, spoof=0.1, bona=0.9):
    """'...XXX...' -> per-frame P(bonafide). 'X' is a spoof frame."""
    return np.array([spoof if c == "X" else bona for c in spec], dtype=float)


def times(n, offset=0.0):
    return offset + np.arange(n) * FRAME_HOP_S


# ---------------------------------------------------------------------------

def test_a_run_becomes_one_range_with_the_expected_edges():
    scores = frames("." * 100 + "X" * 50 + "." * 50)
    t = times(len(scores))
    segs = localizer()._segments(scores, t, np.ones(len(scores), bool),
                                 min_segment_s=0.3, duration_s=4.0)
    assert len(segs) == 1
    # frames 100..149 inclusive -> 2.00 s to 3.00 s
    assert segs[0]["start_s"] == 2.0
    assert segs[0]["end_s"] == 3.0
    assert abs(segs[0]["duration_s"] - 1.0) < 1e-9


def test_short_runs_are_dropped_by_the_duration_gate():
    scores = frames("." * 50 + "X" * 5 + "." * 50)          # 0.10 s
    t = times(len(scores))
    segs = localizer()._segments(scores, t, np.ones(len(scores), bool),
                                 min_segment_s=0.3, duration_s=2.1)
    assert segs == []


def test_two_runs_split_by_a_brief_dip_are_reported_as_one_edit():
    """Two frames scoring just over the threshold in the middle of a splice do
    not make it two splices — and without merging, each half here would fall
    under the 0.3 s duration gate and the whole edit would vanish."""
    scores = frames("." * 20 + "X" * 12 + ".." + "X" * 12 + "." * 20)
    t = times(len(scores))
    segs = localizer()._segments(scores, t, np.ones(len(scores), bool),
                                 min_segment_s=0.3, duration_s=1.32,
                                 merge_gap_s=0.10)
    assert len(segs) == 1
    assert abs(segs[0]["duration_s"] - 0.52) < 1e-9


def test_runs_split_by_a_long_real_passage_stay_separate():
    scores = frames("." * 10 + "X" * 30 + "." * 40 + "X" * 30 + "." * 10)
    t = times(len(scores))
    segs = localizer()._segments(scores, t, np.ones(len(scores), bool),
                                 min_segment_s=0.3, duration_s=2.4,
                                 merge_gap_s=0.10)
    assert len(segs) == 2


def test_non_speech_frames_cannot_form_a_segment():
    """Silence scores as confidently spoof on this checkpoint. Masked frames
    must neither start a run nor extend one."""
    scores = frames("X" * 100)                    # everything looks spoof
    speech = np.zeros(100, dtype=bool)
    t = times(100)
    segs = localizer()._segments(scores, t, speech, min_segment_s=0.3,
                                 duration_s=2.0)
    assert segs == []


def test_a_pause_inside_a_spoof_region_splits_the_reported_ranges():
    """Long non-speech gaps are excluded, so two spoof phrases either side of
    a real pause are two ranges — not one range spanning the pause."""
    scores = frames("X" * 100)
    speech = np.ones(100, dtype=bool)
    speech[40:60] = False                          # 0.4 s pause
    t = times(100)
    segs = localizer()._segments(scores, t, speech, min_segment_s=0.3,
                                 duration_s=2.0, merge_gap_s=0.10)
    assert len(segs) == 2


def test_times_come_from_the_frame_time_array_not_the_index():
    """The chunk-drift fix: a segment in a later chunk is timestamped against
    that chunk's own sample offset."""
    scores = frames("." * 10 + "X" * 30 + "." * 10)
    t = times(len(scores), offset=20.0)            # second 20 s chunk
    segs = localizer()._segments(scores, t, np.ones(len(scores), bool),
                                 min_segment_s=0.3, duration_s=40.0)
    assert segs[0]["start_s"] == 20.2
    assert segs[0]["end_s"] == 20.8


def test_end_time_is_clamped_to_the_file_duration():
    scores = frames("." * 10 + "X" * 30)
    t = times(len(scores))
    segs = localizer()._segments(scores, t, np.ones(len(scores), bool),
                                 min_segment_s=0.3, duration_s=0.75)
    assert segs[0]["end_s"] == 0.75


def test_confidence_is_mean_spoof_probability_over_the_range():
    scores = np.concatenate([np.full(10, 0.9), np.full(30, 0.2)])
    t = times(len(scores))
    segs = localizer()._segments(scores, t, np.ones(len(scores), bool),
                                 min_segment_s=0.3, duration_s=0.8)
    assert abs(segs[0]["confidence"] - 0.8) < 1e-4


# ---------------------------------------------------------------------------

def test_merge_close_runs_is_a_no_op_when_disabled():
    runs = [(0, 5), (6, 10)]
    assert _merge_close_runs(runs, 0) == runs


def test_merge_close_runs_chains_across_several_runs():
    assert _merge_close_runs([(0, 5), (6, 10), (11, 15)], 3) == [(0, 15)]
