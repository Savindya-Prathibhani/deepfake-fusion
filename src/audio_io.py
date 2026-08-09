"""
audio_io.py
-----------
Single entry point for turning any user-supplied file into the one waveform
representation both components consume: mono float32 at 16 kHz.

Why one shared loader for two models
------------------------------------
Component 1 (AASIST) and Component 2 (BAM/WavLM) were both trained at
16 kHz mono. They differ in *framing* (C1 crops/pads to a fixed 64600-sample
window; C2 runs full-length), not in sample rate or channel layout. So the
decode/resample step is shared and the framing stays inside each component's
own inference module, which is what keeps the two models independent.

Video input is handled by extracting the audio track with ffmpeg before
anything else runs, so from `load_audio()` onward nothing downstream needs to
know whether the source was a .mp4 or a .wav.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
TARGET_SR = SAMPLE_RATE

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".wma"}
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".flv", ".wmv"}


class AudioLoadError(RuntimeError):
    pass


def ffmpeg_binary() -> str:
    """Path to an ffmpeg executable.

    Prefers a system ffmpeg; falls back to the one bundled with the
    `imageio-ffmpeg` wheel so the project works on a clean machine without
    the user having to install ffmpeg separately.
    """
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover
        raise AudioLoadError(
            "ffmpeg not found. Install it system-wide, or `pip install imageio-ffmpeg`."
        ) from exc


def is_video(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


def extract_audio_from_video(video_path: str | Path, out_wav: str | Path | None = None) -> Path:
    """Extract the audio track of a video to a 16 kHz mono WAV.

    Resampling is done by ffmpeg here rather than in numpy later because
    ffmpeg's resampler is higher quality than a naive one and this is the
    only place a non-16k source is guaranteed to appear.
    """
    video_path = Path(video_path)
    if out_wav is None:
        out_wav = Path(tempfile.mkdtemp(prefix="dff_")) / (video_path.stem + ".wav")
    out_wav = Path(out_wav)

    cmd = [
        ffmpeg_binary(), "-y", "-i", str(video_path),
        "-vn",                      # drop video
        "-ac", "1",                 # mono
        "-ar", str(TARGET_SR),      # 16 kHz
        "-f", "wav", "-acodec", "pcm_s16le",
        str(out_wav),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out_wav.is_file():
        tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
        raise AudioLoadError(f"ffmpeg failed to extract audio from {video_path.name}:\n{tail}")
    return out_wav


def load_audio(path: str | Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Any audio or video file -> (mono float32 waveform, sample_rate).

    Tries soundfile first (fast, native for wav/flac/ogg), then librosa
    (which routes mp3/m4a/etc. through audioread/ffmpeg). Stereo is
    downmixed by averaging channels, matching the demo path both projects
    were validated with.
    """
    path = Path(path)
    if not path.is_file():
        raise AudioLoadError(f"File not found: {path}")

    if is_video(path):
        path = extract_audio_from_video(path)

    wav = None
    try:
        import soundfile as sf

        wav, sr = sf.read(str(path))
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
    except Exception:
        wav = None

    if wav is None:
        try:
            import librosa

            wav, sr = librosa.load(str(path), sr=None, mono=True)
        except Exception as exc:
            raise AudioLoadError(f"Could not decode {path.name}: {exc}") from exc

    wav = np.asarray(wav, dtype=np.float32)
    if wav.size == 0:
        raise AudioLoadError(f"{path.name} decoded to an empty waveform (no audio track?).")

    if sr != target_sr:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return wav.astype(np.float32), sr
