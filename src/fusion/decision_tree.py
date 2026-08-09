"""
fusion/decision_tree.py
-----------------------
The Fusion Engine: a rule-based decision tree over the two components'
outputs.

The tree implemented here is the specified one:

    Start
      |
    Component 1 - fake confidence?
      |
      +-- High ----------------+-- segment ratio High  -> Fully AI-Generated
      |                        +-- segment ratio Small -> Partially Tampered
      |                        +-- segment ratio None  -> Fully AI-Generated
      |
      +-- Low / Medium --------+-- segment ratio None  -> Real
                               +-- segment ratio Small -> Partially Tampered
                               +-- segment ratio High  -> Fully AI-Generated

The first split is on Component 1's *confidence band*, not on its raw
binary verdict. That distinction is the whole point of the diagram and it is
also the fix for the most common false positive: a genuine recording that
scores just below Component 1's 0.7142 threshold produces a decision margin
of a few thousandths. Treating that identically to a confidently synthetic
file (margin ~1.0) labelled real audio "Fully AI-Generated" on evidence the
model itself was almost undecided about. A Low/Medium band with no localized
spoof region now resolves to Real, which is what the diagram says.

Why a decision tree rather than probability averaging
-----------------------------------------------------
Averaging assumes the two scores are commensurable. They are not:

  * They answer different questions. Component 1 asks "is this whole file
    synthetic?"; Component 2 asks "which frames of this file are synthetic?".
    A partially tampered file is genuinely fake to Component 2 and, depending
    on where the tampering falls, may be genuinely real to Component 1.
  * They are on different, uncalibrated scales. Component 1's decision
    threshold is 0.7142 and its XAI analysis found its softmax saturated;
    Component 2's utterance threshold was 0.0049 on dev and 0.868 on eval -
    an order of magnitude apart across two splits of the same dataset.
    Averaging numbers whose thresholds differ this much is arithmetic without
    meaning.
  * Three-way output. Averaging two binary scores cannot produce
    "Partially Tampered" as a distinct class; it can only produce a point on a
    real-fake line.

Each rule below fires on a stated condition, and the fired rule is returned
with the result, so any verdict can be traced back to the evidence that
produced it.

Every threshold is supplied by the backend (`configs/fusion.yaml` via
`pipeline.DeepfakePipeline`). Nothing in the UI may set one; the defaults in
`DEFAULTS` exist only so `fuse()` remains callable in isolation by the tests.

The two gates on what counts as a spoof region
----------------------------------------------
A segment must clear BOTH a duration floor (`min_segment_s`, applied inside
Component 2) and a confidence floor (`min_segment_confidence`, applied here)
before it counts toward the tampered ratio. The confidence floor is the second
half of the false-positive fix: Component 2 emits a run of frames whenever
P(bonafide) dips below 0.55, and on real audio breaths, silence and codec
artefacts produce exactly such dips - but at a mean P(spoof) barely over 0.5,
not at the 0.9+ a real edit produces. Counting those as tampering turned clean
recordings into "Partially Tampered". Segments below the floor are still
reported in `rejected_segments` so the evidence is not silently discarded.
"""

from __future__ import annotations

REAL = "Real"
FULLY_FAKE = "Fully AI-Generated"
PARTIAL = "Partially Tampered"

# Confidence band for Component 1, and the ratio bands for Component 2.
HIGH = "High"
LOW_MED = "Low/Medium"
RATIO_NONE = "None"
RATIO_SMALL = "Small"
RATIO_HIGH = "High"

# Fallbacks only. The running application always passes the backend config.
DEFAULTS = {
    # Component 1 decision margin at or above which the fake call is "High
    # confidence". 0.50 corresponds to P(bonafide) <= 0.357 against the
    # 0.7142 threshold - comfortably synthetic, not a borderline score.
    "c1_high_confidence": 0.50,
    # Mean P(spoof) a segment must reach before it counts as tampering.
    "min_segment_confidence": 0.70,
    # Flagged fraction at or above which the file is "fully" generated.
    "full_ratio": 0.90,
    # Flagged fraction below which the localization evidence is treated as
    # absent rather than small.
    "min_ratio": 0.05,
}

RULE_TEXT = {
    "R1": "Component 1 is highly confident the whole file is synthetic and "
          "Component 2 flagged essentially all of it.",
    "R2": "Component 1 is highly confident the file is synthetic, but "
          "Component 2 localized the spoof evidence to part of it only - a "
          "splice rather than a wholly generated file.",
    "R3": "Component 1 is highly confident the file is synthetic and "
          "Component 2 found no internal real/fake boundary to localize, "
          "which is what uniform synthesis looks like.",
    "R4": "Component 2 localized no qualifying spoof region and Component 1 "
          "gave no confident evidence of synthesis.",
    "R5": "Component 2 localized one or more spoof regions inside a file "
          "Component 1 read as largely genuine.",
    "R6": "Component 2 flagged essentially the entire file, so this is not a "
          "localized edit whatever Component 1's margin was.",
}


# The tree itself, as a table: (Component 1 band, Component 2 ratio band) ->
# (rule, verdict, which evidence the reported confidence comes from). Every
# leaf of the diagram is one row here, including the two the diagram leaves
# implicit - High/None and Low-Medium/High - so no combination falls through.
TREE = {
    (HIGH,    RATIO_HIGH):  ("R1", FULLY_FAKE, "both"),
    (HIGH,    RATIO_SMALL): ("R2", PARTIAL,    "segments"),
    (HIGH,    RATIO_NONE):  ("R3", FULLY_FAKE, "c1"),
    (LOW_MED, RATIO_NONE):  ("R4", REAL,       "c1_real"),
    (LOW_MED, RATIO_SMALL): ("R5", PARTIAL,    "segments"),
    (LOW_MED, RATIO_HIGH):  ("R6", FULLY_FAKE, "segments_only"),
}


def _c1_band(c1_fake: bool, c1_conf: float, high_confidence: float) -> str:
    """The diagram's first split: High vs Low/Medium fake confidence."""
    return HIGH if (c1_fake and c1_conf >= high_confidence) else LOW_MED


def _ratio_band(has_segments: bool, ratio: float, min_ratio: float,
                full_ratio: float) -> str:
    """The diagram's second split: None / Small / High segment ratio."""
    if not has_segments or ratio < min_ratio:
        return RATIO_NONE
    return RATIO_HIGH if ratio >= full_ratio else RATIO_SMALL


def _confidence(source: str, c1_conf: float, seg_conf: float,
                c1_fake: bool) -> tuple[float, str]:
    """Reported confidence and the one-line statement of what it measures.

    Confidence always comes from the evidence the fired rule actually relied
    on. Where the components disagree it is NOT inflated by pretending they
    agreed - see the `segments_only` row.
    """
    if source == "both":
        return max(seg_conf, c1_conf), \
            "Component 1 decision margin and Component 2 frame coverage"
    if source == "segments":
        return seg_conf, "Mean spoof probability across the reported segments"
    if source == "segments_only":
        return seg_conf, \
            "Component 2 frame coverage (Component 1 was not confident)"
    if source == "c1_real":
        if c1_fake:
            # C1 said fake but only just; the confidence in "Real" is the
            # confidence that its fake call was noise, not the fake margin.
            return 1.0 - c1_conf, (
                "Component 1 called the file fake with a low margin and "
                "Component 2 found nothing to localize")
        return c1_conf, "Component 1 decision margin"
    return c1_conf, "Component 1 decision margin"


def fuse(c1: dict, c2: dict, params: dict | None = None) -> dict:
    """Combine Component 1 and Component 2 outputs into one verdict.

    c1: output of AASISTDetector.predict
    c2: output of BAMLocalizer.predict
    params: backend thresholds, normally `settings["fusion"]` loaded from
            configs/fusion.yaml. Unknown keys are ignored.
    """
    p = dict(DEFAULTS)
    p.update({k: v for k, v in (params or {}).items() if k in DEFAULTS})

    c1_fake = bool(c1["is_fake"])
    c1_conf = float(c1["confidence"])
    duration_s = _duration(c2)

    kept, rejected = _qualifying_segments(c2, p["min_segment_confidence"])
    tampered_s = sum(s["duration_s"] for s in kept)
    ratio = (tampered_s / duration_s) if duration_s > 0 else 0.0
    seg_conf = _mean_segment_confidence(kept)

    # ---- band the two inputs, then read the leaf off the table ---------
    c1_band = _c1_band(c1_fake, c1_conf, p["c1_high_confidence"])
    ratio_band = _ratio_band(bool(kept), ratio, p["min_ratio"], p["full_ratio"])

    rule, label, confidence_source = TREE[(c1_band, ratio_band)]
    confidence, basis = _confidence(confidence_source, c1_conf, seg_conf, c1_fake)

    agreement = (c1_fake and label in (FULLY_FAKE, PARTIAL)) or \
                (not c1_fake and label == REAL)

    result = {
        "final_prediction": label,
        "rule_fired": rule,
        "rule_description": RULE_TEXT[rule],
        "confidence": round(float(confidence), 4),
        "confidence_basis": basis,
        "components_agree": bool(agreement),
        "c1_confidence_band": c1_band,
        "segment_ratio_band": ratio_band,
        "evidence": {
            "c1_score_bonafide": round(float(c1["score_bonafide"]), 6),
            "c1_threshold": float(c1["threshold"]),
            "c1_verdict": c1["verdict"],
            "c1_confidence": round(c1_conf, 4),
            "c2_utt_score_bonafide": round(float(c2["utt_score_bonafide"]), 6),
            "c2_frame_threshold": float(c2["frame_threshold"]),
            "duration_s": round(duration_s, 3),
            "qualifying_segments": len(kept),
            "rejected_segments": len(rejected),
            "tampered_ratio": round(ratio, 6),
            "tampered_duration_s": round(float(tampered_s), 3),
        },
        "params": p,
    }

    # Timestamps are only meaningful for the partial verdict; including them
    # under a "Real" or "Fully AI-Generated" label would invite reading them
    # as localized edits, which is exactly what those labels deny.
    if label == PARTIAL:
        result["tampered_segments"] = kept
        result["tampered_ranges"] = [format_range(s) for s in kept]
        result["tampered_duration_s"] = round(float(tampered_s), 3)
        result["tampered_ratio"] = round(ratio, 6)

    if rejected:
        result["rejected_segments"] = rejected

    return result


def format_range(segment: dict) -> str:
    """One segment as `start:end` in seconds, e.g. `2.23:2.90`.

    This is the reporting format the integration spec asks for. Seconds with
    two decimals, not mm:ss - the segments this system reports are often under
    a second long and a mm:ss rendering hides that.
    """
    return f"{segment['start_s']:.2f}:{segment['end_s']:.2f}"


def format_ranges(segments: list[dict]) -> str:
    """`2.23:2.90 , 3.89:4.90` - the whole list on one line."""
    return " , ".join(format_range(s) for s in segments)


def _duration(c2: dict) -> float:
    """The duration the tampered ratio is a fraction OF.

    Component 2's `analysis_duration_s` - the speech it actually examined -
    rather than the file's wall-clock length. Dividing by wall-clock length
    would make a verdict depend on how much silence someone left at the head
    of the file.
    """
    if c2.get("analysis_duration_s"):
        return float(c2["analysis_duration_s"])
    if c2.get("duration_s"):
        return float(c2["duration_s"])
    # Older/synthetic dicts may not carry it; recover it from the ratio.
    ratio = float(c2.get("tampered_ratio") or 0.0)
    if ratio > 0:
        return float(c2.get("tampered_duration_s", 0.0)) / ratio
    return 0.0


def _qualifying_segments(c2: dict, min_confidence: float
                         ) -> tuple[list[dict], list[dict]]:
    """Split Component 2's segments into those that count and those that do not.

    Component 2 has already applied the duration floor; this applies the
    confidence floor. Both lists are returned so a rejected segment appears in
    the JSON rather than vanishing.
    """
    kept, rejected = [], []
    for s in (c2.get("segments") or []):
        (kept if float(s["confidence"]) >= min_confidence else rejected).append(s)
    return kept, rejected


def _mean_segment_confidence(segments: list[dict]) -> float:
    """Duration-weighted mean of the per-segment spoof confidence.

    Duration-weighted rather than a plain mean so a single long, confidently
    spoof region is not dragged down by a short marginal one.
    """
    total = sum(s["duration_s"] for s in segments)
    if total <= 0:
        return 0.0
    return sum(s["confidence"] * s["duration_s"] for s in segments) / total
