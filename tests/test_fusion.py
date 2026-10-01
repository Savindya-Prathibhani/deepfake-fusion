"""
Unit tests for the Fusion Engine.

These run without torch, without the checkpoints and without audio — the
fusion engine takes two dictionaries and nothing else, which is exactly what
makes it testable in isolation. Run with:  python -m pytest tests/ -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fusion.decision_tree import (   # noqa: E402
    FULLY_FAKE, PARTIAL, REAL, format_ranges, fuse,
)

C1_THRESHOLD = 0.7142295241350909


def c1(score, threshold=C1_THRESHOLD):
    """Mirrors AASISTDetector.predict's output, including its margin rule."""
    is_fake = score < threshold
    conf = (threshold - score) / threshold if is_fake else (score - threshold) / (1 - threshold)
    return {
        "score_bonafide": score, "threshold": threshold,
        "verdict": "FAKE" if is_fake else "REAL", "is_fake": is_fake,
        "confidence": min(1.0, conf),
    }


def c2(segments, duration=10.0, utt=0.5, frame_thr=0.55):
    tampered = sum(s["duration_s"] for s in segments)
    return {
        "utt_score_bonafide": utt, "frame_threshold": frame_thr,
        "segments": segments, "n_segments": len(segments),
        "duration_s": duration,
        "tampered_duration_s": tampered,
        "tampered_ratio": tampered / duration if duration else 0.0,
    }


def seg(start, end, confidence=0.9):
    return {"start_s": start, "end_s": end, "duration_s": end - start,
            "confidence": confidence}


# ---------------------------------------------------------------------------
# The six leaves of the specified tree
# ---------------------------------------------------------------------------

def test_r1_high_c1_confidence_and_high_ratio():
    r = fuse(c1(0.02), c2([seg(0.0, 9.8)], duration=10.0))
    assert r["final_prediction"] == FULLY_FAKE
    assert r["rule_fired"] == "R1"


def test_r2_high_c1_confidence_but_localized_ratio():
    """C1 sure it is fake, C2 says only part of it — a splice, not a whole."""
    r = fuse(c1(0.02), c2([seg(3.0, 5.5)], duration=10.0))
    assert r["final_prediction"] == PARTIAL
    assert r["rule_fired"] == "R2"


def test_r3_uniform_synthesis_has_no_internal_boundary():
    """A fully generated file gives Component 2 nothing to localize."""
    r = fuse(c1(0.02), c2([]))
    assert r["final_prediction"] == FULLY_FAKE
    assert r["rule_fired"] == "R3"
    assert "tampered_segments" not in r


def test_r4_clean_real_file():
    r = fuse(c1(0.98), c2([]))
    assert r["final_prediction"] == REAL
    assert r["rule_fired"] == "R4"
    assert r["components_agree"]


def test_r5_localized_edit_in_an_otherwise_real_file():
    r = fuse(c1(0.9), c2([seg(3.0, 5.5)], duration=10.0))
    assert r["final_prediction"] == PARTIAL
    assert r["rule_fired"] == "R5"
    assert r["tampered_segments"] == [seg(3.0, 5.5)]
    assert abs(r["tampered_ratio"] - 0.25) < 1e-9


def test_r6_everything_flagged_is_not_partial():
    """C1 unconvinced, but C2 flagged the whole file — not a localized edit."""
    r = fuse(c1(0.99), c2([seg(0.0, 9.9)], duration=10.0))
    assert r["final_prediction"] == FULLY_FAKE
    assert r["rule_fired"] == "R6"
    assert not r["components_agree"]


# ---------------------------------------------------------------------------
# The two false-positive fixes
# ---------------------------------------------------------------------------

def test_marginal_c1_fake_score_alone_does_not_condemn_a_file():
    """The regression that made real audio read as fake.

    A genuine recording scoring 0.70 against the 0.7142 threshold is 0.014
    from the line — a margin of 0.02. Component 2 found nothing. The old tree
    fired R1 and returned "Fully AI-Generated" at 2% confidence; the specified
    tree puts this in the Low/Medium band, where "no segments" means Real.
    """
    r = fuse(c1(0.70), c2([]))
    assert r["final_prediction"] == REAL
    assert r["rule_fired"] == "R4"
    assert r["c1_confidence_band"] == "Low/Medium"


def test_integration_uses_c1_score_when_boolean_flag_is_stale():
    """The score/threshold pair is authoritative at the fusion boundary."""
    c1_output = c1(0.02)
    c1_output["is_fake"] = False
    c1_output["verdict"] = "REAL"

    r = fuse(c1_output, c2([]))

    assert r["final_prediction"] == FULLY_FAKE
    assert r["rule_fired"] == "R3"


def test_weakly_flagged_region_in_real_audio_is_not_tampering():
    """Breaths and codec artefacts drift past the frame threshold at ~0.5
    P(spoof); a real splice sits at 0.9+. Only the latter counts."""
    r = fuse(c1(0.95), c2([seg(2.0, 3.4, confidence=0.52)], duration=10.0))
    assert r["final_prediction"] == REAL
    assert r["rule_fired"] == "R4"
    # Rejected, not hidden.
    assert len(r["rejected_segments"]) == 1
    assert r["evidence"]["qualifying_segments"] == 0


def test_confident_short_region_still_counts():
    """The confidence gate must not simply suppress every short segment."""
    r = fuse(c1(0.95), c2([seg(2.23, 2.90, confidence=0.94)], duration=10.0))
    assert r["final_prediction"] == PARTIAL
    assert r["evidence"]["qualifying_segments"] == 1


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def test_time_ranges_are_reported_in_the_specified_format():
    r = fuse(c1(0.9), c2([seg(2.23, 2.90), seg(3.89, 4.90)], duration=10.0))
    assert r["final_prediction"] == PARTIAL
    assert r["tampered_ranges"] == ["2.23:2.90", "3.89:4.90"]
    assert format_ranges(r["tampered_segments"]) == "2.23:2.90 , 3.89:4.90"


def test_ranges_are_absent_from_non_partial_verdicts():
    """Timestamps under a Real or Fully-Generated label would invite reading
    them as localized edits, which is what those labels deny."""
    for r in (fuse(c1(0.98), c2([])), fuse(c1(0.02), c2([]))):
        assert "tampered_ranges" not in r
        assert "tampered_segments" not in r


# ---------------------------------------------------------------------------
# Engine properties
# ---------------------------------------------------------------------------

def test_thresholds_come_from_the_caller_not_from_the_ui():
    """Backend params must actually reach the tree — the same evidence under a
    stricter confidence gate has to change the verdict."""
    evidence = (c1(0.95), c2([seg(2.0, 4.0, confidence=0.80)], duration=10.0))
    assert fuse(*evidence)["final_prediction"] == PARTIAL
    strict = fuse(*evidence, params={"min_segment_confidence": 0.90})
    assert strict["final_prediction"] == REAL
    assert strict["params"]["min_segment_confidence"] == 0.90


def test_unknown_config_keys_are_ignored():
    """`settings["fusion"]` also carries min_segment_s, which belongs to
    Component 2, not to the tree."""
    r = fuse(c1(0.98), c2([]), params={"min_segment_s": 0.3, "nonsense": 1})
    assert "min_segment_s" not in r["params"]
    assert r["final_prediction"] == REAL


def test_confidence_is_duration_weighted_not_averaged():
    """A long confident segment must not be dragged down by a short weak one."""
    r = fuse(c1(0.9), c2([seg(0.0, 8.0, 0.95), seg(9.0, 9.2, 0.75)], duration=20.0))
    assert r["final_prediction"] == PARTIAL
    assert r["confidence"] > 0.93          # plain mean would give 0.85


def test_no_averaging_of_component_scores():
    """Identical component scores must still give different verdicts when the
    localization evidence differs — proof the engine is not score-averaging."""
    a = fuse(c1(0.3), c2([], duration=10.0))
    b = fuse(c1(0.3), c2([seg(2.0, 4.0)], duration=10.0))
    assert a["final_prediction"] == FULLY_FAKE
    assert b["final_prediction"] == PARTIAL


def test_tiny_flagged_fraction_in_a_long_file_is_noise():
    """A single 1.5 s flag in a 2-minute file, with C1 saying real, is 1.25% —
    below the min_ratio floor, so the ratio band is None."""
    r = fuse(c1(0.95), c2([seg(30.0, 31.5)], duration=120.0))
    assert r["final_prediction"] == REAL
    assert r["segment_ratio_band"] == "None"
