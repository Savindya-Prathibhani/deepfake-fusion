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

Decoder order, and why ffmpeg is second rather than last
--------------------------------------------------------
`soundfile` (libsndfile) is tried first: it is fast and native for WAV, FLAC
and OGG. It cannot open AAC in an MP4 container, which is what a `.m4a` is —
every iPhone voice memo, every WhatsApp voice note. libsndfile has no AAC
decoder and reports `Format not recognised`.

librosa is NOT the fallback for those. Modern librosa delegates to soundfile
first and only reaches `audioread` for what soundfile refuses, and audioread is
deprecated, ships no backend of its own, and raises `NoBackendError` with an
empty message — which is why an unreadable `.m4a` previously surfaced to the
user as the bare string "Could not decode file.m4a:" with nothing after the
colon.

ffmpeg is therefore the second decoder, not the last. It is already a hard
dependency of this project for video, it is already bundled through
`imageio-ffmpeg` so it is present on a clean machine, and it decodes every
format the uploader accepts. librosa is kept as a third attempt only because it
costs nothing to try before giving up.
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


def decode_with_ffmpeg(path: str | Path, out_wav: str | Path | None = None,
                       target_sr: int = TARGET_SR) -> Path:
    """Any media file -> a 16 kHz mono PCM WAV that soundfile can always read.

    Used for video (extracting the audio track) and for every audio format
    libsndfile cannot open itself, which is what `.m4a`/AAC is.

    Resampling is done by ffmpeg here rather than in numpy later because
    ffmpeg's resampler is higher quality than a naive one, and because doing it
    at decode time means a container with an unusual rate never reaches the
    rest of the pipeline.
    """
    path = Path(path)
    if out_wav is None:
        out_wav = Path(tempfile.mkdtemp(prefix="dff_")) / (path.stem + ".wav")
    out_wav = Path(out_wav)

    cmd = [
        ffmpeg_binary(), "-y", "-i", str(path),
        "-vn",                      # drop any video stream
        "-ac", "1",                 # mono
        "-ar", str(target_sr),      # 16 kHz
        "-f", "wav", "-acodec", "pcm_s16le",
        str(out_wav),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out_wav.is_file():
        tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
        raise AudioLoadError(f"ffmpeg could not decode {path.name}:\n{tail}")
    return out_wav


def extract_audio_from_video(video_path: str | Path,
                             out_wav: str | Path | None = None) -> Path:
    """Extract the audio track of a video to a 16 kHz mono WAV."""
    return decode_with_ffmpeg(video_path, out_wav)


def _read_with_soundfile(path: Path) -> tuple[np.ndarray, int]:
    """Stereo is downmixed by averaging channels, matching the demo path both
    projects were validated with."""
    import soundfile as sf

    wav, sr = sf.read(str(path))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav, sr


def load_audio(path: str | Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Any audio or video file -> (mono float32 waveform, sample_rate).

    Decoders are tried in the order documented at the top of this module:
    soundfile, then ffmpeg, then librosa. Whichever succeeds, the waveform
    that comes back is mono float32 at `target_sr`.
    """
    path = Path(path)
    if not path.is_file():
        raise AudioLoadError(f"File not found: {path}")

    temp_dir: Path | None = None
    if is_video(path):
        decoded = decode_with_ffmpeg(path, target_sr=target_sr)
        temp_dir = decoded.parent
        path = decoded

    attempts: list[str] = []
    wav = sr = None

    try:
        wav, sr = _read_with_soundfile(path)
    except Exception as exc:
        attempts.append(f"soundfile: {exc}")

    # The .m4a path. libsndfile has no AAC decoder, so this is not a fallback
    # for an exotic file - it is the normal route for one of the commonest
    # formats a phone produces.
    if wav is None:
        try:
            decoded = decode_with_ffmpeg(path, target_sr=target_sr)
            temp_dir = temp_dir or decoded.parent
            wav, sr = _read_with_soundfile(decoded)
        except Exception as exc:
            attempts.append(f"ffmpeg: {exc}")

    if wav is None:
        try:
            import warnings

            import librosa

            # This path only runs when the file is already known to be
            # undecodable, so librosa's "PySoundFile failed, trying audioread"
            # and audioread's own deprecation notice are noise on top of an
            # error the caller is about to be told about properly.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wav, sr = librosa.load(str(path), sr=None, mono=True)
        except Exception as exc:
            # audioread raises NoBackendError with an empty message, which on
            # its own tells the user nothing at all.
            attempts.append(f"librosa: {exc or type(exc).__name__}")

    if temp_dir is not None:
        shutil.rmtree(temp_dir, ignore_errors=True)

    if wav is None:
        detail = "\n  ".join(attempts)
        raise AudioLoadError(
            f"Could not decode {path.name}. Every decoder failed:\n  {detail}")

    wav = np.asarray(wav, dtype=np.float32)
    if wav.size == 0:
        raise AudioLoadError(f"{path.name} decoded to an empty waveform (no audio track?).")

    if sr != target_sr:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return wav.astype(np.float32), sr
