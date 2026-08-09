"""
component2_bam/infer.py
-----------------------
Inference-only wrapper around H1-Enhanced BAM (Component 2): whole-utterance
score plus frame-level spoof localization.

What was reused, verbatim
-------------------------
  * `model_setup.py`   - copied unchanged from the research project. It defines
                         BAM, the Boundary Enhancement module, Boundary
                         Framewise Attention, `build_boundary_features` (the H1
                         change), and `build_model` / `build_feature_extractor`
                         / `preprocess_batch`.
  * `localization.py`  - copied unchanged. Only `frames_to_segments` is used
                         here, but the file is kept whole so the segmentation
                         rule cannot drift from the one that produced the
                         reported segment-F1 numbers.
  * The scoring rule below - identical to `evaluate._forward_pass_over_split`:
                         `softmax(seg_logits, dim=2)[0, :, 1]` per frame and
                         `softmax(utt_logits, dim=1)[:, 1]` per utterance, both
                         P(bonafide).

What was dropped, and why
-------------------------
  * `evaluate.py` itself imports `train`, `matplotlib`, `seaborn` and `sklearn`
    to compute EER, confusion matrices and figures over a labelled split. None
    of that exists at inference time - there are no labels for a user's file.
    The ~15 lines of `_forward_pass_over_split` that actually touch the model
    are reproduced above instead, which is what removes the entire training
    stack from the dependency graph.
  * `dataset.py` / `labels.py` load ground-truth segment labels from the
    PartialSpoof annotation files. Inference has no labels, so both are out.
  * `utils.load_checkpoint` is replaced by a three-line `torch.load` here to
    avoid pulling in the config-schema/validation/status machinery, which is a
    training-pipeline concern.

Architecture is NOT hardcoded
-----------------------------
The model is built from `configs/bam_h1_boundary_gated_seghead.yaml` through
the project's own `model_setup.build_model`, so `boundary_gated_seg_head: true`
(and every other flag) comes from the same config the checkpoint was trained
under. The checkpoint's `seg_head.1.weight` is 2 x 259 = 2 x (256 + 3), which
only loads cleanly if the three H1 boundary features are wired in - so a config
mismatch fails loudly at `load_state_dict` rather than silently scoring wrong.

Long-input handling - a deliberate, documented deviation
--------------------------------------------------------
Evaluation ran full-length utterances (a few seconds each). WavLM self-attention
is quadratic in sequence length, so a 5-minute file would exhaust memory. Inputs
longer than `chunk_seconds` are therefore split into consecutive chunks, each
run independently, and their frame scores concatenated. Chunk length is a whole
number of 320-sample WavLM hops so frame indices stay aligned to absolute time.
Files shorter than the chunk length take the original single-pass path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import localization      # vendored, unchanged
import model_setup       # vendored, unchanged

SAMPLE_RATE = 16000
FRAME_HOP_S = 0.02       # WavLM stride: 320 samples @16 kHz
WAVLM_HOP_SAMPLES = 320


class BAMLocalizer:
    """Frame-level spoof localizer. Independent of Component 1."""

    def __init__(self, config: dict, checkpoint_path: str | Path,
                 frame_threshold: float, utt_threshold: float | None = None,
                 device: str | None = None, chunk_seconds: float = 20.0):
        self.config = config
        self.device = device or model_setup.get_device()
        self.frame_threshold = float(frame_threshold)
        self.utt_threshold = None if utt_threshold is None else float(utt_threshold)

        # Whole number of WavLM hops, so chunk k's frame f maps to a stable
        # absolute time and segment boundaries do not drift across chunks.
        n_hops = int(round(chunk_seconds * SAMPLE_RATE / WAVLM_HOP_SAMPLES))
        self.chunk_samples = n_hops * WAVLM_HOP_SAMPLES

        # build_model resolves its own device via get_device(); move it again
        # so an explicit `device` argument (e.g. --device cpu on a CUDA box)
        # is actually honoured rather than silently ignored.
        self.model = model_setup.build_model(config).to(self.device)
        ckpt = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        self.model.load_state_dict(state)
        self.model.eval()

        self.feature_extractor = model_setup.build_feature_extractor(config)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _forward_one(self, wav: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        """One chunk -> (utt P(bonafide), per-frame P(bonafide), boundary probs)."""
        wav_t = torch.from_numpy(np.asarray(wav, dtype=np.float32))
        input_values = model_setup.preprocess_batch(
            [wav_t], self.feature_extractor, SAMPLE_RATE, self.device)

        utt_logits, seg_logits, boundary_logits = self.model(input_values)

        utt_score = float(F.softmax(utt_logits, dim=1)[0, 1].item())
        frame_scores = F.softmax(seg_logits, dim=2)[0, :, 1].float().cpu().numpy()
        boundary_probs = torch.sigmoid(boundary_logits)[0].float().cpu().numpy()
        return utt_score, frame_scores, boundary_probs

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, wav: np.ndarray, min_segment_s: float = 1.0,
                speech_mask=None, merge_gap_s: float = 0.10) -> dict:
        """wav: mono float32 @16 kHz.

        Frame scores are P(bonafide): a frame is called spoof when its score
        is BELOW `frame_threshold`. That direction is fixed by
        `localization._target_binary` and must not be inverted.

        `speech_mask`: optional `speech_gate.SpeechMask`. Non-speech frames are
        excluded from spoof runs and from the denominator of the tampered
        ratio. BAM was trained on PartialSpoof utterances and has no defined
        behaviour on silence; this checkpoint scores six seconds of digital
        silence at a mean P(spoof) of 0.88, higher than many genuine splices,
        so the pauses in an ordinary recording would otherwise be reported as
        tampered regions. The ratio is taken over speech duration rather than
        file duration for the same reason: a file that is half silence and
        half wholly synthetic speech is fully generated, not 50% tampered.
        """
        duration_s = len(wav) / SAMPLE_RATE
        if duration_s <= 0:
            raise ValueError("Empty waveform: nothing to analyse.")

        utt_scores, frame_parts, bdy_parts, time_parts = [], [], [], []

        for start in range(0, len(wav), self.chunk_samples):
            chunk = wav[start:start + self.chunk_samples]
            if start > 0 and len(chunk) < WAVLM_HOP_SAMPLES * 25:   # < 0.5 s tail
                break
            u, f, b = self._forward_one(chunk)
            utt_scores.append(u)
            frame_parts.append(f)
            bdy_parts.append(b)
            # Absolute start time of every frame in this chunk. WavLM's conv
            # front end emits floor((L - 400)/320) + 1 frames for L samples,
            # i.e. one FEWER than L/320, so simply concatenating chunks and
            # multiplying the index by the hop drifts ~20 ms earlier per
            # chunk - 1.2 s of error at the end of a 20-minute file, enough
            # to put a reported timestamp on the wrong word. Anchoring each
            # chunk to its own sample offset removes the drift entirely.
            chunk_start_s = start / SAMPLE_RATE
            time_parts.append(chunk_start_s + np.arange(len(f)) * FRAME_HOP_S)

        frame_scores = np.concatenate(frame_parts)
        boundary_probs = np.concatenate(bdy_parts)
        frame_times = np.concatenate(time_parts)
        # Chunk-level utterance scores are combined by taking the minimum:
        # the utterance head answers "is any of this spoof", so one confidently
        # spoof chunk should not be diluted by many clean ones.
        utt_score = float(np.min(utt_scores))

        if speech_mask is not None:
            is_speech = speech_mask.at(frame_times)
            # The denominator is the speech actually covered by analysed
            # frames, not the mask's total, so a dropped sub-0.5 s tail chunk
            # cannot make the ratio exceed 1.
            analysis_duration_s = float(is_speech.sum()) * FRAME_HOP_S
        else:
            is_speech = np.ones(frame_scores.shape, dtype=bool)
            analysis_duration_s = duration_s

        segments = self._segments(frame_scores, frame_times, is_speech,
                                  min_segment_s, duration_s, merge_gap_s)
        tampered_s = float(sum(s["duration_s"] for s in segments))
        denom = analysis_duration_s if analysis_duration_s > 0 else duration_s

        return {
            "utt_score_bonafide": utt_score,
            "utt_threshold": self.utt_threshold,
            "frame_threshold": self.frame_threshold,
            "frame_hop_s": FRAME_HOP_S,
            "n_frames": int(frame_scores.size),
            "n_speech_frames": int(is_speech.sum()),
            "duration_s": round(duration_s, 3),
            # What the tampered ratio is a fraction OF. The fusion engine uses
            # this, not duration_s.
            "analysis_duration_s": round(analysis_duration_s, 3),
            "segments": segments,
            "n_segments": len(segments),
            "tampered_duration_s": round(tampered_s, 3),
            "tampered_ratio": round(tampered_s / denom, 6) if denom > 0 else 0.0,
            "min_segment_s": min_segment_s,
            "merge_gap_s": merge_gap_s,
            "_frame_scores": frame_scores,        # kept for the UI plot
            "_frame_times_s": frame_times,        # kept for the UI plot
            "_is_speech": is_speech,              # kept for the UI plot
            "_boundary_probs": boundary_probs,
        }

    # ------------------------------------------------------------------
    def _segments(self, frame_scores: np.ndarray, frame_times: np.ndarray,
                  is_speech: np.ndarray, min_segment_s: float,
                  duration_s: float, merge_gap_s: float = 0.0) -> list[dict]:
        """Frame scores -> spoof segments in seconds, short ones discarded.

        Segmentation delegates to the project's own `frames_to_segments`, so
        the contiguous-run definition here is the same one the reported
        segment-F1 was measured with.

        Segments shorter than `min_segment_s` are dropped, per the integration
        spec. Note what this costs: PartialSpoof's spoof spans are frequently
        under a second, so this filter removes real detections as well as
        spurious ones. It is a product decision about what is worth reporting
        to a user, not a claim that sub-second detections are wrong. The
        second gate - a minimum mean spoof probability - lives in the fusion
        engine, so this stays a pure frames-to-seconds conversion.

        Times come from `frame_times`, not from index * hop, so a segment in
        the fifth chunk of a long file is timestamped against its own chunk's
        sample offset rather than against an index that has drifted.

        Runs separated by less than `merge_gap_s` are joined before the
        duration filter. A splice does not stop being one splice because two
        frames in the middle of it scored 0.56; reporting it as two ranges
        would overstate how many edits were found, and each fragment might
        then fall under the duration floor and be discarded entirely.
        """
        # A non-speech frame is not evidence either way, so it neither starts a
        # run nor extends one. Clearing it here rather than post-filtering the
        # segments also means a pause between two spoof phrases splits them
        # into two reported ranges instead of one range that spans the pause.
        pred_spoof = ((frame_scores < self.frame_threshold) & is_speech).astype(int)
        runs = localization.frames_to_segments(pred_spoof, pos_value=1)
        runs = _merge_close_runs(runs, int(round(merge_gap_s / FRAME_HOP_S)))

        out = []
        for start_f, end_f in runs:
            start_s = float(frame_times[start_f])
            # end_f is exclusive: the segment ends one hop after its last frame.
            end_s = min(float(frame_times[end_f - 1]) + FRAME_HOP_S, duration_s)
            dur = end_s - start_s
            if dur < min_segment_s:
                continue
            seg_scores = frame_scores[start_f:end_f]
            out.append({
                "start_s": round(float(start_s), 3),
                "end_s": round(float(end_s), 3),
                "duration_s": round(float(dur), 3),
                # Mean P(spoof) across the segment's frames.
                "confidence": round(float(np.mean(1.0 - seg_scores)), 4),
            })
        return out


def _merge_close_runs(runs: list[tuple[int, int]], max_gap_frames: int
                      ) -> list[tuple[int, int]]:
    """Join half-open frame runs separated by fewer than `max_gap_frames`."""
    if max_gap_frames <= 0 or len(runs) < 2:
        return runs
    merged = [runs[0]]
    for start, end in runs[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end < max_gap_frames:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged
