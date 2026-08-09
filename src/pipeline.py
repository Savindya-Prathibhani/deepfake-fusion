"""
pipeline.py
-----------
Orchestration: load once, then for each file run Component 1 and Component 2
independently and hand only their OUTPUTS to the Fusion Engine.

The independence is structural, not just stylistic. `AASISTDetector` and
`BAMLocalizer` never see each other; `fuse()` receives two plain dicts and
imports neither model. Either component can be swapped, retrained, or removed
without touching the other two files.
"""

from __future__ import annotations

import time
from pathlib import Path

import yaml

from audio_io import SAMPLE_RATE, is_video, load_audio
from component1_aasist.infer import AASISTDetector
from component2_bam.infer import BAMLocalizer
from fusion.decision_tree import fuse
from speech_gate import SpeechMask

ROOT = Path(__file__).resolve().parent.parent


def load_settings(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "fusion.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


class DeepfakePipeline:
    """Both detectors plus the fusion engine, loaded once and reused."""

    def __init__(self, settings: dict | None = None, device: str | None = None):
        self.settings = settings or load_settings()
        s = self.settings

        c1 = s["component1"]
        self.c1 = AASISTDetector(
            checkpoint_path=_resolve(c1["checkpoint"]),
            threshold=c1["threshold"],
            device=device,
            aggregate=c1.get("window_aggregate", "mean"),
        )

        c2 = s["component2"]
        with open(_resolve(c2["config"])) as f:
            bam_config = yaml.safe_load(f)
        self.c2 = BAMLocalizer(
            config=bam_config,
            checkpoint_path=_resolve(c2["checkpoint"]),
            frame_threshold=c2["frame_threshold"],
            utt_threshold=c2.get("utt_threshold"),
            device=device,
            chunk_seconds=c2.get("chunk_seconds", 20.0),
        )

    def analyze(self, file_path: str | Path) -> dict:
        file_path = Path(file_path)
        t0 = time.time()

        wav, sr = load_audio(file_path)
        t_load = time.time()

        # Computed once, from the waveform alone, and handed to both
        # components. It is signal processing, not a third model, so the two
        # detectors stay independent of each other and of anything learned.
        gate_cfg = self.settings.get("speech_gate", {})
        mask = SpeechMask.from_waveform(wav, sr, gate_cfg)

        c1_out = self.c1.predict(
            wav, speech_mask=mask,
            min_speech_fraction=float(gate_cfg.get("min_window_speech", 0.5)))
        t_c1 = time.time()

        c2_out = self.c2.predict(
            wav, min_segment_s=self.settings["fusion"]["min_segment_s"],
            speech_mask=mask,
            merge_gap_s=float(self.settings["fusion"].get("merge_gap_s", 0.10)))
        t_c2 = time.time()

        fusion_out = fuse(c1_out, c2_out, self.settings["fusion"])

        # A file with essentially no speech is outside both models' domain.
        # The verdict is still reported - suppressing it would be its own kind
        # of lie - but it is reported next to the reason not to trust it.
        if mask.speech_ratio < float(gate_cfg.get("min_file_speech_ratio", 0.05)):
            fusion_out.setdefault("warnings", []).append(
                f"Only {mask.speech_ratio*100:.1f}% of this file was detected "
                "as speech. Both components were trained exclusively on "
                "speech; treat this verdict as unreliable."
            )

        # Underscore-prefixed arrays are for the UI plot only; they are large
        # and not part of the JSON contract.
        c2_json = {k: v for k, v in c2_out.items() if not k.startswith("_")}

        return {
            "input": {
                "filename": file_path.name,
                "source_type": "video" if is_video(file_path) else "audio",
                "duration_s": round(len(wav) / sr, 3),
                "sample_rate": sr,
            },
            "speech_gate": mask.summary(),
            "component1_band_augmented_aasist": c1_out,
            "component2_h1_enhanced_bam": c2_json,
            "fusion": fusion_out,
            "timing_s": {
                "load": round(t_load - t0, 3),
                "component1": round(t_c1 - t_load, 3),
                "component2": round(t_c2 - t_c1, 3),
                "total": round(time.time() - t0, 3),
            },
            "_frame_scores": c2_out["_frame_scores"],
            "_frame_times_s": c2_out["_frame_times_s"],
            "_is_speech": c2_out["_is_speech"],
            "_frame_hop_s": c2_out["frame_hop_s"],
        }


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p
