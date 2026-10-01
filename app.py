"""
app.py — Streamlit UI for the audio deepfake detector.

Run:  streamlit run app.py

The interface deliberately presents one detector, not two components and a
fusion engine. A viewer is being asked "is this recording genuine?", and the
architecture that answers it is not part of that question: naming the models,
showing which rule fired, or reporting each model's separate score invites the
reader to second-guess the verdict with numbers they have no way to weigh.
Everything needed to audit a result is still produced — it goes into the
downloadable JSON record, which carries the full component-level evidence.
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
from pipeline import DeepfakePipeline   # noqa: E402

st.set_page_config(page_title="Audio Deepfake Detector", layout="wide")

VERDICT_STYLE = {
    "Real": ("#1a7f37", "✔"),
    "Fully AI-Generated": ("#b42318", "✖"),
    "Partially Tampered": ("#b54708", "▲"),
}

# What the viewer is told, per verdict. Written from the listener's point of
# view rather than the engine's: no rule numbers, no model names, no bands.
VERDICT_TEXT = {
    "Real": "No synthetic or edited speech was detected in this recording.",
    "Fully AI-Generated":
        "This recording appears to be synthetic speech throughout, rather than "
        "a genuine recording with edits inserted into it.",
    "Partially Tampered":
        "This recording is largely genuine, but one or more passages appear to "
        "be synthetic. The affected time ranges are listed below.",
}


@st.cache_resource(show_spinner="Loading the detector (first run downloads its "
                                "speech model)…")
def get_pipeline():
    return DeepfakePipeline()


def main():
    st.title("Audio Deepfake Detection")
    st.caption(
        "Detects fully AI-generated speech and audio that is genuine except "
        "for inserted manipulated passages, which it locates in time."
    )

    with st.sidebar:
        render_about()

    uploaded = st.file_uploader(
        "Upload audio or video",
        type=["wav", "flac", "mp3", "m4a", "ogg", "opus", "aac", "wma",
              "mp4", "mkv", "mov", "avi", "webm"],
    )
    if uploaded is None:
        st.info("Upload a file to begin. Video files have their audio track "
                "extracted automatically.")
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


def render_about():
    """What the tool does and what it cannot do. No thresholds.

    The operating point is fixed in `configs/fusion.yaml`; it is neither shown
    nor adjustable here. A detector whose thresholds a viewer can read off and
    argue with is inviting exactly the second-guessing that a stated verdict
    exists to settle, and one they can drag has no reportable operating point
    at all.
    """
    st.header("About")
    st.markdown(
        "Upload a recording and the detector returns one of three verdicts:\n\n"
        "- **Real** — no synthetic speech found\n"
        "- **Partially Tampered** — genuine, with manipulated passages spliced "
        "in; their time ranges are reported\n"
        "- **Fully AI-Generated** — synthetic throughout\n"
    )
    st.divider()
    st.caption(
        "Analysis covers the speech in a recording; music, silence and room "
        "tone are not evidence either way and are excluded.\n\n"
        "Confidence is a decision margin, not a calibrated probability. "
        "Results on phone recordings, social-media audio and heavily "
        "compressed uploads have not been validated — treat those as "
        "indicative."
    )


def render(result: dict):
    fusion = result["fusion"]
    label = fusion["final_prediction"]
    colour, mark = VERDICT_STYLE[label]

    st.markdown(
        f"<div style='padding:1.1rem 1.3rem;border-radius:10px;"
        f"background:{colour}14;border-left:6px solid {colour};'>"
        f"<div style='font-size:1.7rem;font-weight:650;color:{colour};'>"
        f"{mark}&nbsp;{label}</div>"
        f"<div style='opacity:.75;margin-top:.35rem;'>"
        f"{VERDICT_TEXT[label]}</div></div>",
        unsafe_allow_html=True,
    )

    ev = fusion["evidence"]
    segments = fusion.get("tampered_segments") or []

    st.write("")
    a, b, c = st.columns(3)
    a.metric("Confidence", f"{fusion['confidence']*100:.1f}%")
    b.metric("Duration", f"{result['input']['duration_s']:.2f} s")
    if label == "Partially Tampered":
        c.metric("Manipulated passages", len(segments))
    else:
        c.metric("Synthetic speech", f"{ev['tampered_ratio']*100:.1f}%")

    for message in fusion.get("warnings", []):
        st.warning(message)

    if not fusion["components_agree"]:
        st.warning(
            "The evidence for this verdict is mixed: the recording as a whole "
            "and the individual passages within it point in different "
            "directions. Treat the result as less certain than the confidence "
            "figure suggests."
        )

    if label == "Partially Tampered" and segments:
        render_segments(segments, ev, result["input"]["duration_s"])

    render_timeline(result)

    payload = {k: v for k, v in result.items() if not k.startswith("_")}
    st.download_button(
        "Download full report (JSON)",
        json.dumps(payload, indent=2),
        file_name=f"{Path(result['input']['filename']).stem}_result.json",
        mime="application/json",
        help="The complete technical record of this analysis, for review.",
    )


def render_segments(segments: list[dict], evidence: dict, duration_s: float):
    st.subheader("Manipulated passages")
    st.code(format_ranges(segments), language=None)
    st.dataframe(
        [
            {
                "Range (s)": format_range(s),
                "Start (s)": f"{s['start_s']:.2f}",
                "End (s)": f"{s['end_s']:.2f}",
                "Duration (s)": f"{s['duration_s']:.2f}",
                "Confidence": f"{s['confidence']*100:.1f}%",
            }
            for s in segments
        ],
        use_container_width=True, hide_index=True,
    )
    st.caption(
        f"{evidence['tampered_duration_s']:.2f} s of synthetic speech in a "
        f"{duration_s:.2f} s recording."
    )


def render_timeline(result: dict):
    """Where in the recording the audio looks synthetic.

    Kept because it is the one internal quantity a listener can actually check:
    the peaks line up with passages they can play back, which is a different
    thing from being handed a model's score and asked to trust it.
    """
    scores = result.get("_frame_scores")
    if scores is None or not len(scores):
        return

    st.subheader("Where the recording looks synthetic")
    p_synthetic = 1.0 - np.asarray(scores, dtype=float)
    times = np.asarray(result["_frame_times_s"], dtype=float)

    # A 5-minute file is ~15,000 points and a 40-minute one ~120,000; handing
    # that many to the browser stalls it. Downsample by taking the MAX of each
    # bucket, not the mean, so a short synthetic burst stays visible instead of
    # being averaged away.
    max_points = 2000
    if len(p_synthetic) > max_points:
        bucket = int(np.ceil(len(p_synthetic) / max_points))
        pad = (-len(p_synthetic)) % bucket
        padded = np.concatenate([p_synthetic, np.full(pad, np.nan)])
        plotted = np.nanmax(padded.reshape(-1, bucket), axis=1)
        plotted_t = times[::bucket][:len(plotted)]
    else:
        plotted = p_synthetic
        plotted_t = times

    # x is real time in seconds, so a reported range like 2.23:2.90 can be read
    # straight off this chart. Plotting against the frame index instead would
    # make the two disagree on any file long enough to be analysed in chunks.
    st.line_chart(
        {"Time (s)": plotted_t.tolist(),
         "Likelihood of synthesis": plotted.tolist()},
        x="Time (s)", y="Likelihood of synthesis",
    )

    excluded = 1.0 - float(result.get("speech_gate", {}).get("speech_ratio", 1.0))
    caption = ("Higher means the audio at that moment resembles synthetic "
               "speech. Peaks alone are not a verdict — a passage is only "
               "reported above once it is sustained and strong enough.")
    if excluded > 0.001:
        caption += (f" {excluded*100:.0f}% of this file is silence or "
                    "non-speech and was excluded from the verdict; readings "
                    "over those stretches are not meaningful.")
    st.caption(caption)


if __name__ == "__main__":
    main()
