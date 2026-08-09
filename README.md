# Audio Deepfake Detection — Integrated Application

Two independently trained detectors plus a rule-based fusion engine, behind one
Streamlit UI and one CLI. Accepts audio or video; video has its audio track
extracted automatically.

| | Component 1 | Component 2 |
|---|---|---|
| Model | Band-Augmented AASIST | H1-Enhanced BAM |
| Front end | Sinc convolution (raw waveform) | WavLM-base-plus (raw waveform) |
| Question | Is the **whole file** synthetic? | **Which frames** are synthetic? |
| Output | P(bonafide), one score | P(bonafide) per 20 ms frame + boundary probabilities |
| Trained on | ASVspoof 2019 LA | PartialSpoof v1.2 |
| Reported | 5.19% EER (test) | 0.909 segment F1 @ IoU ≥ 0.5 (dev) |

Neither model imports the other. The Fusion Engine imports neither — it receives
two plain dictionaries.

---

## 1. Setup

```bash
git clone <your-repo> deepfake-fusion
cd deepfake-fusion

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Place the two checkpoints

They are not in the repository (385 MB + 3.7 MB). Copy them in:

```
models/aasist/best.pt      <- Component 1, trained Band-Augmented AASIST
models/bam/best.pth        <- Component 2, trained H1-Enhanced BAM
```

Nothing else needs to be downloaded manually. On first run, `transformers`
fetches `microsoft/wavlm-base-plus` from Hugging Face (~380 MB, cached in
`~/.cache/huggingface`) because `model_setup.BAM` builds its backbone with
`WavLMModel.from_pretrained` before the checkpoint weights are loaded over it.
The first launch therefore needs an internet connection; later launches do not.

---

## 2. Run

```bash
streamlit run app.py          # UI
python run_cli.py sample.mp4  # JSON to stdout, one summary line to stderr
python run_cli.py sample.wav --out result.json
python run_cli.py sample.wav --quiet | jq .fusion.tampered_ranges
python -m pytest tests/ -q    # fusion, speech gate and segmentation; no models needed
```

The stderr summary is one line per file:

```
interview.wav: Partially Tampered (91.4%, R5)  tampered: 2.23:2.90 , 3.89:4.90
```

In VS Code: open the folder, select `.venv` as the interpreter
(`Ctrl+Shift+P` → *Python: Select Interpreter*), then run the commands above in
the integrated terminal. `.vscode/launch.json` is included, so **F5** runs the
Streamlit app directly.

---

## 3. Structure

```
deepfake-fusion/
├── app.py                      Streamlit UI
├── run_cli.py                  headless CLI, same JSON
├── configs/
│   ├── fusion.yaml             thresholds + fusion parameters (all provenance documented)
│   └── bam_h1_boundary_gated_seghead.yaml   Component 2 architecture config, unmodified
├── models/
│   ├── aasist/best.pt
│   └── bam/best.pth
└── src/
    ├── audio_io.py             video → audio, decode, resample to 16 kHz mono
    ├── speech_gate.py          energy VAD; confines both models to speech
    ├── pipeline.py             loads both detectors, runs them, calls fusion
    ├── component1_aasist/
    │   ├── infer.py            AASIST inference wrapper
    │   └── external/           vendored clovaai/aasist source (MIT) + AASIST.conf
    ├── component2_bam/
    │   ├── infer.py            BAM inference wrapper
    │   ├── model_setup.py      vendored unchanged from the research project
    │   └── localization.py     vendored unchanged (frames → segments)
    └── fusion/
        └── decision_tree.py    the Fusion Engine
```

### What was removed from the research projects

Everything training- or analysis-only: `train.py`, `dataset.py`, `labels.py`,
`losses.py`, `bootstrap.py`, `utils.py` (config schema / hashing / status),
`gradcam.py`, `perturbation.py`, `embeddings.py`, `feature_attribution.py`,
`statistics.py`, `bootstrap.py`, `segment_eval.py`, `boundary_candidates.py`,
`localization_operating_point.py`, `evaluate.py` from both projects, and every
notebook. None of it is reachable at inference time.

Two files were vendored *unchanged* rather than rewritten — `model_setup.py`
and `localization.py` — because they define the architecture and the
segmentation rule that produced the reported numbers. Rewriting either would
put the app on a different definition from the thesis results.

---

## 4. How a decision is made

```
                        file (audio or video)
                                 │
                    16 kHz mono float32 waveform
                                 │
                    speech gate (energy VAD, 20 ms)
                    ┌────────────┴────────────┐
          Component 1                    Component 2
     4.04 s windows                  20 s chunks, 20 ms frames
     non-speech windows dropped      non-speech frames dropped
     P(bonafide) per window          P(bonafide) per frame
            │                                │
   mean → whole-file score        frames < threshold → runs → seconds
   margin vs 0.7142 → band        merge, drop < 0.30 s, drop weak
            └────────────┬───────────────────┘
                    Fusion Engine
              (C1 confidence band × C2 ratio band)
```

The tree is the specified one, and it is a table rather than an ordered rule
list — every combination of the two bands has exactly one leaf:

| | ratio **None** | ratio **Small** | ratio **High** |
|---|---|---|---|
| **C1 fake confidence High** | R3 Fully AI-Generated | R2 Partially Tampered | R1 Fully AI-Generated |
| **C1 fake confidence Low/Medium** | R4 Real | R5 Partially Tampered | R6 Fully AI-Generated |

* **C1 band.** *High* means C1 called the file fake **and** its decision margin
  reached `c1_high_confidence` (0.50). Otherwise *Low/Medium*.
* **C2 band.** *None* is no qualifying segment, or a tampered ratio below
  `min_ratio` (0.05); *High* is at or above `full_ratio` (0.90); *Small* is
  everything between. The ratio is a fraction of the **analysed speech**, not of
  the file's wall-clock length.
* A segment qualifies only if it is at least `min_segment_s` (0.30 s) long
  **and** its mean P(spoof) reaches `min_segment_confidence` (0.70). Segments
  that fail the second gate are still listed under `rejected_segments`.

**Why rules and not probability averaging.** The two scores are not
commensurable. They answer different questions, their thresholds sit an order of
magnitude apart (0.714 vs 0.0049–0.868), Component 1's softmax is known from its
own XAI analysis to be saturated, and averaging two binary scores cannot produce
a third class. Each verdict instead names the rule that fired, so it can be
traced to the evidence.

**R3 is the rule that needs the pair.** A uniformly synthetic file has no
internal real→fake boundary, so Component 2's boundary module has nothing to
localize and may return no segments at all. Component 1 carries that case.
R1/R6 are the mirror: if nearly every speech frame is flagged, the file is not
*partially* anything.

**Why the first split is on the band and not the verdict.** Component 1's
threshold is 0.7142. A genuine recording scoring 0.70 is 0.014 from the line — a
margin of 0.02 — and treating that as equivalent to a score of 0.02 is what
produced "Fully AI-Generated" on real audio. In the Low/Medium band, Component 1
alone can no longer condemn a file; it needs Component 2 to localize something.

### Reported time ranges

A `Partially Tampered` verdict carries `tampered_ranges`, in `start:end`
seconds:

```
2.23:2.90 , 3.89:4.90
```

The same values appear as `tampered_segments` with per-segment durations and
confidences, in the UI table, and on the CLI's stderr summary line.

---

## 4a. The speech gate

Both components were trained exclusively on speech utterances and neither has
defined behaviour on silence. Measured on the two checkpoints in `models/`:

| input | C1 P(bonafide) | C2 mean P(spoof) per frame |
|---|---|---|
| digital silence, 6 s | 0.317 → **FAKE** | 0.880 |
| white noise, 6 s | 0.737 | 0.938 |

Both call silence spoof, and Component 2 does so more confidently than it calls
many genuine splices. Every real recording contains pauses and room tone, so
without a gate those regions become "tampered segments" and the pauses drag
Component 1's windowed mean below its threshold — real audio comes back fake.

The gate is a relative-energy VAD on the 20 ms grid (95th-percentile reference,
−35 dB relative, −55 dBFS absolute floor, 0.20 s gap bridging, 0.10 s hangover).
It is signal processing, not a third model, so the two detectors remain
independent of each other and of anything learned. It changes neither model — it
confines both to the domain they were measured on. Turn it off with
`speech_gate.enabled: false` to reproduce pre-gate behaviour.

If less than `min_file_speech_ratio` of a file is speech, the verdict is still
reported but carries an explicit warning that it is out of domain.

---

## 5. Thresholds and where they come from

**Every threshold lives in `configs/fusion.yaml` and nowhere else.** The UI
displays them read-only; it has no controls that change them. A detector whose
operating point moves when a viewer drags a slider has no reportable operating
point — two people would get two verdicts on the same file and neither could be
cited. Edit the config, where the change is versioned next to its provenance.

| Threshold | Value | Source |
|---|---|---|
| C1 decision | 0.7142295 | `band_augment/metrics/test_results.json` → `val_eer_threshold_used` (validation-derived, no test leakage) |
| C1 high-confidence margin | 0.50 | Fusion parameter: P(bonafide) ≤ 0.357 against the C1 threshold |
| C2 frame decision | 0.55 | Segment-F1 sweep over `eval_artifacts_dev.npz` at IoU ≥ 0.5 |
| Minimum segment | 0.30 s | Fusion parameter (was 1.0 s — see below) |
| Merge gap | 0.10 s | Fusion parameter |
| Minimum segment confidence | 0.70 | Fusion parameter |
| Ratio "none" below | 0.05 | Fusion parameter |
| Ratio "high" at or above | 0.90 | Fusion parameter |
| Speech gate | −35 dB rel. / −55 dBFS abs. | Measured out-of-domain behaviour, §4a |

### On the minimum segment length

It was 1.0 s. PartialSpoof's spoof spans are frequently well under a second, so
a 1 s floor discarded most genuine partial-tampering detections and made the
reported ranges unrepresentative of what the model actually found. 0.30 s is
~15 frames at the 20 ms frame rate — long enough that a short noisy run cannot
survive, short enough to keep real edits. The `merge_gap_s` join runs first, so
a splice broken by two frames scoring just over the threshold is measured, and
reported, as the one region it is.

### On the Component 2 frame threshold

The value stored as `frame_threshold` inside `eval_artifacts_*.npz` is the frame
**EER** threshold — 0.7024 on dev, 0.9985 on eval — not a segment-F1 selection.
Sweeping dev for segment F1 at IoU ≥ 0.5 gives:

| threshold | seg P | seg R | seg F1 |
|---|---|---|---|
| 0.50 | 0.9110 | 0.8996 | 0.9053 |
| **0.55** | 0.9125 | 0.9040 | **0.9082** |
| 0.70 | 0.9068 | 0.9097 | 0.9083 |
| 0.9985 | 0.5144 | 0.5822 | 0.5462 |

The dev optimum is flat between 0.55 and 0.70 (0.9082 vs 0.9083 — a tie).
`0.55` is chosen because it was also the peak on the **eval** split, where the
EER-derived threshold lands at 0.9985 and collapses segment F1 to 0.546. Of two
statistically tied dev choices, the one that also holds under distribution shift
is the better default. Change it in `configs/fusion.yaml`.

---

## 6. Documented deviations from the validated pipelines

Four, all forced by accepting arbitrary real-world input where evaluation used
short, labelled, speech-only utterances.

0. **The speech gate** (§4a). Evaluation never presented either model with a
   non-speech frame. Real uploads are mostly not like that.

1. **Component 1, long files.** Evaluation centre-crops to 64600 samples.
   Centre-cropping a 3-minute file would discard most of it, so long inputs are
   split into consecutive 4.04 s windows and the window scores averaged. Files
   ≤ 4.04 s take the original single-window path unchanged.
2. **Component 2, long files.** Evaluation ran full-length utterances of a few
   seconds. WavLM attention is quadratic in length, so inputs are split into
   20 s chunks (a whole number of 320-sample hops, so frame→time alignment does
   not drift) and their frame scores concatenated. Utterance scores across
   chunks are combined by minimum, not mean.
3. **The segment floor.** Any duration floor discards genuine detections as
   well as spurious ones; it is a decision about what is worth showing a user,
   not a claim that shorter detections are wrong. It is now 0.30 s rather than
   1.0 s, and is set in `configs/fusion.yaml`, not in the UI.

## 7. Honest limits

Component 1 scored 5.19% EER in-domain (ASVspoof 2019 LA) but 42.61% EER on
In-The-Wild audio — near chance. Component 2 was trained and measured on
PartialSpoof. Neither has been validated on social-media audio, phone
recordings, or compressed uploads. Treat output on such files as illustrative.

Confidence figures are decision margins, not calibrated probabilities.

## 8. Attribution

`src/component1_aasist/external/` contains the AASIST reference implementation
from [clovaai/aasist](https://github.com/clovaai/aasist), commit
`a04c9863f63d44471dde8a6abcb3b082b07cd1d1`, MIT licence (see
`AASIST_LICENSE`). Component 2's backbone is
[`microsoft/wavlm-base-plus`](https://huggingface.co/microsoft/wavlm-base-plus).
