"""
localization.py
----------------
Localization-specific metrics, kept deliberately separate from
detection metrics (evaluate.py) per standing project instruction:
"a frame can be correctly classified while its predicted boundaries are
inaccurate" - detection accuracy and localization accuracy must always be
reported and reasoned about separately.

Fixes two bugs found in the original notebook's segment-IoU code:

  BUG 2 (target class was implicit and wrong): the original code measured
  IoU on whatever frames_to_segments() treated as "1", which was the
  BONAFIDE class (since scores were P(bonafide) and labels used 1=bonafide).
  Every function here takes an explicit `target_class` argument
  ("spoof" | "bonafide") so this can never silently happen again.

  BUG 3 (precision/recall reported for the wrong class by default): not
  this module's concern directly (that's evaluate.py's frame-level
  metrics), but the same discipline - explicit class labeling - is applied
  throughout.

All functions operate on lists of per-utterance arrays (frame scores and
frame labels are inherently per-utterance; flattening loses segment
boundaries and would silently corrupt IoU/onset-offset calculations across
utterance boundaries).
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

TargetClass = Literal["spoof", "bonafide"]


# ============================================================================
# Basic segment utilities
# ============================================================================

def frames_to_segments(binary_frames: np.ndarray, pos_value: int = 1) -> list[tuple[int, int]]:
    """Contiguous runs of `pos_value` in a 1D binary/int array, as
    half-open (start_idx, end_idx) frame-index pairs.
    """
    segments = []
    in_seg = False
    start = 0
    for i, v in enumerate(binary_frames):
        if v == pos_value and not in_seg:
            start, in_seg = i, True
        if v != pos_value and in_seg:
            segments.append((start, i))
            in_seg = False
    if in_seg:
        segments.append((start, len(binary_frames)))
    return segments


def iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def _target_binary(scores: np.ndarray, labels: np.ndarray, threshold: float,
                    target_class: TargetClass) -> tuple[np.ndarray, np.ndarray]:
    """scores: P(bonafide) per frame (the model's native output convention).
    labels: 1=bonafide, 0=spoof per frame.
    Returns (pred_target_binary, true_target_binary), both 1 = "this frame
    IS target_class", regardless of which class that is - this is what lets
    every downstream function use pos_value=1 uniformly instead of juggling
    which raw label value means what.
    """
    pred_bonafide = (scores >= threshold).astype(int)
    true_bonafide = labels.astype(int)
    if target_class == "spoof":
        return 1 - pred_bonafide, 1 - true_bonafide
    elif target_class == "bonafide":
        return pred_bonafide, true_bonafide
    raise ValueError(f"target_class must be 'spoof' or 'bonafide', got {target_class!r}")


def _greedy_match(pred_segs: list, true_segs: list, min_iou: float):
    """Greedy best-IoU-first matching between predicted and true segments.
    A pair is eligible if iou > 0 AND iou >= min_iou (min_iou=0 means "any
    overlap counts"). Each segment matches at most once.

    Returns (matched_pairs, unmatched_pred_idxs, unmatched_true_idxs) where
    matched_pairs is a list of (pred_idx, true_idx, iou_value).
    """
    candidates = []
    for pi, ps in enumerate(pred_segs):
        for ti, ts in enumerate(true_segs):
            v = iou(ps, ts)
            if v > 0 and v >= min_iou:
                candidates.append((v, pi, ti))
    candidates.sort(reverse=True)

    matched_pred, matched_true, matched_pairs = set(), set(), []
    for v, pi, ti in candidates:
        if pi in matched_pred or ti in matched_true:
            continue
        matched_pairs.append((pi, ti, v))
        matched_pred.add(pi)
        matched_true.add(ti)

    unmatched_pred = [i for i in range(len(pred_segs)) if i not in matched_pred]
    unmatched_true = [i for i in range(len(true_segs)) if i not in matched_true]
    return matched_pairs, unmatched_pred, unmatched_true


# ============================================================================
# Segment-level precision/recall/F1/IoU
# ============================================================================

def segment_localization_metrics(
    frame_scores_list: list[np.ndarray], frame_labels_list: list[np.ndarray],
    threshold: float, target_class: TargetClass = "spoof", iou_thresh: float = 0.5,
) -> dict:
    """Segment-level precision/recall/F1 for `target_class` (default:
    spoof, since that's the class whose localization this project cares
    about - BUG 2 fix). A predicted segment counts as a true positive only
    if some true segment overlaps it with IoU >= iou_thresh (greedy
    best-match-first, one-to-one).
    """
    tp, fp, fn, ious = 0, 0, 0, []

    for scores, labels in zip(frame_scores_list, frame_labels_list):
        pred_target, true_target = _target_binary(scores, labels, threshold, target_class)
        pred_segs = frames_to_segments(pred_target, pos_value=1)
        true_segs = frames_to_segments(true_target, pos_value=1)

        matched_pairs, unmatched_pred, unmatched_true = _greedy_match(pred_segs, true_segs, iou_thresh)
        tp += len(matched_pairs)
        fp += len(unmatched_pred)
        fn += len(unmatched_true)
        ious.extend(v for _, _, v in matched_pairs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    seg_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "target_class": target_class,
        "iou_thresh": iou_thresh,
        "segment_precision": precision,
        "segment_recall": recall,
        "segment_f1": seg_f1,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
    }


# ============================================================================
# Onset / offset timing error
# ============================================================================

def onset_offset_errors(
    frame_scores_list: list[np.ndarray], frame_labels_list: list[np.ndarray],
    threshold: float, target_class: TargetClass = "spoof", hop_s: float = 0.02,
) -> dict:
    """For every true target-class segment that has ANY overlapping
    predicted segment (IoU > 0, greedy best-match-first - a looser
    criterion than segment_localization_metrics' iou_thresh, since a
    detection can be "roughly in the right place" without clearing 0.5
    IoU, and we still want to know how far off its edges are), report the
    onset (start) and offset (end) timing error in seconds.

    True segments with NO overlapping prediction at all are reported
    separately as `n_missed` (localization total failure, not a timing
    error). Predicted segments with no overlapping true segment are
    `n_false_alarm`.
    """
    onset_errors, offset_errors = [], []
    n_missed, n_false_alarm = 0, 0

    for scores, labels in zip(frame_scores_list, frame_labels_list):
        pred_target, true_target = _target_binary(scores, labels, threshold, target_class)
        pred_segs = frames_to_segments(pred_target, pos_value=1)
        true_segs = frames_to_segments(true_target, pos_value=1)

        matched_pairs, unmatched_pred, unmatched_true = _greedy_match(pred_segs, true_segs, min_iou=0.0)

        for pi, ti, _ in matched_pairs:
            ps, ts = pred_segs[pi], true_segs[ti]
            onset_errors.append(abs(ps[0] - ts[0]) * hop_s)
            offset_errors.append(abs(ps[1] - ts[1]) * hop_s)

        n_missed += len(unmatched_true)
        n_false_alarm += len(unmatched_pred)

    def _stats(arr):
        if not arr:
            return {"mean": None, "median": None, "max": None}
        a = np.array(arr)
        return {"mean": float(a.mean()), "median": float(np.median(a)), "max": float(a.max())}

    return {
        "target_class": target_class,
        "n_matched": len(onset_errors),
        "n_missed": n_missed,
        "n_false_alarm": n_false_alarm,
        "onset_error_s": _stats(onset_errors),
        "offset_error_s": _stats(offset_errors),
    }


# ============================================================================
# Boundary detection F1 @ tolerance
# ============================================================================

def boundary_f1_at_tolerance(
    true_boundary_list: list[np.ndarray], pred_boundary_prob_list: list[np.ndarray],
    tolerance_frames: int, threshold: float = 0.5,
) -> dict:
    """Boundary-detection precision/recall/F1 allowing a tolerance window
    of +/- tolerance_frames between a predicted boundary frame and a true
    boundary frame. Matching is greedy, nearest-first, one-to-one (a true
    boundary can be claimed by at most one predicted boundary and vice
    versa), so tolerance windows can't let one prediction "cover" multiple
    true boundaries for free.
    """
    tp, fp, fn = 0, 0, 0

    for true_bdy, pred_prob in zip(true_boundary_list, pred_boundary_prob_list):
        true_idxs = np.where(np.asarray(true_bdy) == 1)[0]
        pred_idxs = np.where(np.asarray(pred_prob) >= threshold)[0]

        candidates = []
        for pi, p in enumerate(pred_idxs):
            for ti, t in enumerate(true_idxs):
                d = abs(int(p) - int(t))
                if d <= tolerance_frames:
                    candidates.append((d, pi, ti))
        candidates.sort()   # nearest first

        matched_pred, matched_true = set(), set()
        for d, pi, ti in candidates:
            if pi in matched_pred or ti in matched_true:
                continue
            matched_pred.add(pi)
            matched_true.add(ti)

        tp += len(matched_pred)
        fp += len(pred_idxs) - len(matched_pred)
        fn += len(true_idxs) - len(matched_true)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tolerance_frames": tolerance_frames,
        "threshold": threshold,
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
    }


# ============================================================================
# Detection-conditioned localization
# ============================================================================

def detection_conditioned_localization(
    utt_scores: np.ndarray, utt_labels: np.ndarray, utt_threshold: float,
    frame_scores_list: list[np.ndarray], frame_labels_list: list[np.ndarray],
    frame_threshold: float, target_class: TargetClass = "spoof", iou_thresh: float = 0.5,
) -> dict:
    """Restricts segment_localization_metrics to only those spoof
    utterances that were CORRECTLY detected as spoof at the utterance
    level. Separates "the model didn't even flag this as spoof"
    (a detection failure) from "the model flagged it but got the
    boundaries wrong" (a genuine localization failure) - the two
    directly conflate in a plain frame-level EER.
    """
    utt_scores = np.asarray(utt_scores)
    utt_labels = np.asarray(utt_labels)
    pred_bonafide = utt_scores >= utt_threshold

    is_true_spoof = (utt_labels == 0)
    is_correctly_detected_spoof = is_true_spoof & (~pred_bonafide)

    included_idxs = np.where(is_correctly_detected_spoof)[0]
    filtered_scores = [frame_scores_list[i] for i in included_idxs]
    filtered_labels = [frame_labels_list[i] for i in included_idxs]

    metrics = segment_localization_metrics(
        filtered_scores, filtered_labels, frame_threshold,
        target_class=target_class, iou_thresh=iou_thresh,
    )
    metrics["n_true_spoof_utterances"] = int(is_true_spoof.sum())
    metrics["n_correctly_detected_spoof_utterances"] = int(len(included_idxs))
    metrics["detection_rate_among_spoof"] = (
        float(len(included_idxs) / is_true_spoof.sum()) if is_true_spoof.sum() > 0 else None
    )
    return metrics
