"""
app.py — Streamlit UI for the integrated audio deepfake detector.

Run:  streamlit run app.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fusion.decision_tree import format_range, format_ranges   # noqa: E402
from pipeline import DeepfakePipeline, load_settings           # noqa: E402

st.set_page_config(page_title="Audio Deepfake Detector", layout="wide")

VERDICT_STYLE = {
    "Real": ("#1a7f37", "✔"),
    "Fully AI-Generated": ("#b42318", "✖"),
    "Partially Tampered": ("#b54708", "▲"),
}


@st.cache_resource(show_spinner="Loading both models (first run downloads WavLM)…")
def get_pipeline():
    return DeepfakePipeline()


def main():
    st.title("Audio Deepfake Detection")
    st.caption(
        "Component 1 — Band-Augmented AASIST (whole-file) · "
        "Component 2 — H1-Enhanced BAM (frame-level localization) · "
        "rule-based Fusion Engine"
    )

    settings = load_settings()

    with st.sidebar:
        render_thresholds(settings)

    uploaded = st.file_uploader(
        "Upload audio or video",
        type=["wav", "flac", "mp3", "m4a", "ogg", "opus", "aac",
              "mp4", "mkv", "mov", "avi", "webm"],
    )
    if uploaded is None:
        st.info("Upload a file to begin. Video files have their audio track "
                "extracted automatically at 16 kHz mono.")
        return

    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = tmp.name

    pipeline = get_pipeline()

    with st.spinner("Analysing…"):
        try:
            result = pipeline.analyze(tmp_path)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            return

    render(result)


def render_thresholds(settings: dict):
    """Show the operating point. Read-only, deliberately.

    Every one of these values is set in `configs/fusion.yaml` and nowhere
    else. They were selected against held-out data, and a threshold a viewer
    can drag is not an operating point — two people would get two verdicts on
    the same file and neither could be cited. Change them in the config, where
    the change is versioned and its provenance is written down.
    """
    c1, c2, f = settings["component1"], settings["component2"], settings["fusion"]
    g = settings.get("speech_gate", {})

    st.header("Operating point")
    st.caption("Set in `configs/fusion.yaml` — backend only, not adjustable here.")

    st.markdown(
        f"""
| Threshold | Value |
|---|---|
| C1 decision (P(bonafide)) | `{c1['threshold']:.4f}` |
| C1 high-confidence margin | `{f['c1_high_confidence']:.2f}` |
| C2 frame decision | `{c2['frame_threshold']:.2f}` |
| Minimum segment | `{f['min_segment_s']:.2f}` s |
| Merge ranges closer than | `{f.get('merge_gap_s', 0.10):.2f}` s |
| Minimum segment confidence | `{f['min_segment_confidence']:.2f}` |
| Ratio: "none" below | `{f['min_ratio']:.2f}` |
| Ratio: "high" at or above | `{f['full_ratio']:.2f}` |
| Speech gate | `{"on" if g.get("enabled", True) else "off"}` |
"""
    )
    st.caption(
        "C1 threshold: band_augment validation EER threshold. "
        "C2 frame threshold: dev segment-F1 sweep at IoU ≥ 0.5. "
        "The speech gate restricts both components to speech frames; both were "
        "trained only on speech and both score silence as confidently fake."
    )


def render(result: dict):
    fusion = result["fusion"]
    c1 = result["component1_band_augmented_aasist"]
    c2 = result["component2_h1_enhanced_bam"]
    label = fusion["final_prediction"]
    colour, mark = VERDICT_STYLE[label]

    st.markdown(
        f"<div style='padding:1.1rem 1.3rem;border-radius:10px;"
        f"background:{colour}14;border-left:6px solid {colour};'>"
        f"<div style='font-size:1.7rem;font-weight:650;color:{colour};'>"
        f"{mark}&nbsp;{label}</div>"
        f"<div style='opacity:.75;margin-top:.35rem;'>"
        f"{fusion['rule_description']}</div>"
        f"<div style='opacity:.55;margin-top:.5rem;font-size:.85rem;'>"
        f"{fusion['rule_fired']} · Component 1 confidence band: "
        f"{fusion['c1_confidence_band']} · Component 2 segment ratio: "
        f"{fusion['segment_ratio_band']}</div></div>",
        unsafe_allow_html=True,
    )

    st.write("")
    a, b, c, d = st.columns(4)
    a.metric("Overall confidence", f"{fusion['confidence']*100:.1f}%")
    b.metric("Whole-audio confidence (C1)", f"{c1['confidence']*100:.1f}%",
             help=f"P(bonafide) = {c1['score_bonafide']:.4f} vs threshold "
                  f"{c1['threshold']:.4f} → {c1['verdict']}")
    ev = fusion["evidence"]
    c.metric("Tampered ratio", f"{ev['tampered_ratio']*100:.1f}%",
             help="Fraction of the analysed SPEECH covered by segments that "
                  "cleared both the duration and the confidence gate. Silence "
                  "is excluded from the denominator, so the number does not "
                  "depend on how much dead air the file was saved with.")
    d.metric("Fake segments", ev["qualifying_segments"],
             help=f"{ev['rejected_segments']} further segment(s) were flagged "
                  "by Component 2 but fell below the confidence gate.")

    for message in fusion.get("warnings", []):
        st.warning(message)

    if not fusion["components_agree"]:
        st.warning(
            "The two components disagreed. The verdict follows the rule shown "
            "above; treat it as lower-confidence than the number suggests."
        )

    segments = fusion.get("tampered_segments") or []
    if label == "Partially Tampered" and segments:
        st.subheader("Fake segment time ranges")
        st.code(format_ranges(segments), language=None)
        st.dataframe(
            [
                {
                    "Range (s)": format_range(s),
                    "Start (s)": f"{s['start_s']:.2f}",
                    "End (s)": f"{s['end_s']:.2f}",
                    "Duration (s)": f"{s['duration_s']:.2f}",
                    "Segment confidence": f"{s['confidence']*100:.1f}%",
                }
                for s in segments
            ],
            use_container_width=True, hide_index=True,
        )
        st.caption(
            f"Total tampered: {ev['tampered_duration_s']:.2f} s of "
            f"{result['input']['duration_s']:.2f} s "
            f"({ev['tampered_ratio']*100:.1f}%)."
        )

    rejected = fusion.get("rejected_segments") or []
    if rejected:
        with st.expander(
            f"{len(rejected)} region(s) flagged by Component 2 but not counted"
        ):
            st.caption(
                "These cleared the duration floor but their mean spoof "
                f"probability was below "
                f"{fusion['params']['min_segment_confidence']:.2f}. On genuine "
                "audio, breaths, silence and codec artefacts produce exactly "
                "this: a run of weakly flagged frames. They are shown so the "
                "evidence is not hidden, not because they indicate tampering."
            )
            st.dataframe(
                [
                    {
                        "Range (s)": format_range(s),
                        "Duration (s)": f"{s['duration_s']:.2f}",
                        "Segment confidence": f"{s['confidence']*100:.1f}%",
                    }
                    for s in rejected
                ],
                use_container_width=True, hide_index=True,
            )

    scores = result.get("_frame_scores")
    if scores is not None and len(scores):
        st.subheader("Frame-level spoof probability")
        hop = result["_frame_hop_s"]
        p_spoof = 1.0 - np.asarray(scores, dtype=float)
        times = np.asarray(result["_frame_times_s"], dtype=float)

        # A 5-minute file is ~15,000 frames and a 40-minute one ~120,000;
        # handing that many points to the browser stalls it. Downsample by
        # taking the MAX of each bucket, not the mean, so a short spoof burst
        # stays visible instead of being averaged away.
        max_points = 2000
        if len(p_spoof) > max_points:
            bucket = int(np.ceil(len(p_spoof) / max_points))
            pad = (-len(p_spoof)) % bucket
            padded = np.concatenate([p_spoof, np.full(pad, np.nan)])
            plotted = np.nanmax(padded.reshape(-1, bucket), axis=1)
            plotted_t = times[::bucket][:len(plotted)]
            step_s = hop * bucket
        else:
            plotted = p_spoof
            plotted_t = times
            step_s = hop

        # x is real time in seconds, so a reported range like 2.23:2.90 can be
        # read straight off this chart. Plotting against the frame index
        # instead would make the two disagree on any file long enough to be
        # chunked.
        st.line_chart(
            {"Time (s)": plotted_t.tolist(), "P(spoof)": plotted.tolist()},
            x="Time (s)", y="P(spoof)",
        )
        gate = result.get("speech_gate", {})
        excluded = 1.0 - float(gate.get("speech_ratio", 1.0))
        st.caption(
            f"{len(p_spoof):,} frames of {hop*1000:.0f} ms. "
            f"One point on this chart = {step_s*1000:.0f} ms. "
            f"Frames above {1 - c2['frame_threshold']:.2f} P(spoof) are counted "
            "as spoof, before the minimum-duration and minimum-confidence "
            "gates are applied. "
            + (f"{excluded*100:.1f}% of the file was non-speech and is excluded "
               "from the verdict entirely — high P(spoof) over those stretches "
               "is out-of-domain output, not evidence of tampering."
               if excluded > 0.001 else "")
        )

    with st.expander("Component detail"):
        left, right = st.columns(2)
        with left:
            st.markdown("**Component 1 — Band-Augmented AASIST**")
            st.write({k: v for k, v in c1.items() if k != "window_scores"})
        with right:
            st.markdown("**Component 2 — H1-Enhanced BAM**")
            st.write({k: v for k, v in c2.items() if k != "segments"})

    payload = {k: v for k, v in result.items() if not k.startswith("_")}
    with st.expander("JSON output"):
        st.json(payload)
    st.download_button(
        "Download JSON",
        json.dumps(payload, indent=2),
        file_name=f"{Path(result['input']['filename']).stem}_result.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
