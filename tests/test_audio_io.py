"""
Decoder tests.

These build their own fixtures with the bundled ffmpeg rather than committing
binary audio to the repository, and skip cleanly if ffmpeg is unavailable.

`.m4a` is the case that matters: it is AAC in an MP4 container, which is what
every iPhone voice memo and WhatsApp voice note is, and libsndfile cannot open
it. It is tested here alongside the formats that never broke, so a future change
to the decoder chain cannot quietly restore the failure.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from audio_io import AudioLoadError, ffmpeg_binary, load_audio   # noqa: E402

SR = 16000


@pytest.fixture(scope="module")
def ffmpeg():
    try:
        return ffmpeg_binary()
    except AudioLoadError:
        pytest.skip("ffmpeg not available")


@pytest.fixture(scope="module")
def source_wav(tmp_path_factory):
    import soundfile as sf

    t = np.arange(SR * 3) / SR
    wav = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    path = tmp_path_factory.mktemp("audio") / "source.wav"
    sf.write(path, wav, SR)
    return path


def transcode(ffmpeg, source, out_path, *args):
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(source),
                    *args, str(out_path)], check=True, capture_output=True)
    return out_path


@pytest.mark.parametrize("suffix,args", [
    (".m4a", ["-c:a", "aac"]),
    (".mp3", ["-c:a", "libmp3lame"]),
    (".flac", ["-c:a", "flac"]),
    (".ogg", ["-c:a", "libvorbis"]),
])
def test_every_accepted_audio_format_decodes(ffmpeg, source_wav, tmp_path,
                                             suffix, args):
    path = transcode(ffmpeg, source_wav, tmp_path / f"clip{suffix}", *args)
    wav, sr = load_audio(path)
    assert sr == SR
    assert wav.dtype == np.float32
    assert 2.8 < len(wav) / sr < 3.3        # lossy codecs pad by a few frames


def test_m4a_at_a_different_sample_rate_is_resampled(ffmpeg, source_wav, tmp_path):
    """A phone recording is rarely already at 16 kHz. Resampling happens at
    decode time, so nothing downstream ever sees another rate."""
    path = transcode(ffmpeg, source_wav, tmp_path / "clip.m4a",
                     "-c:a", "aac", "-ar", "44100")
    wav, sr = load_audio(path)
    assert sr == SR
    assert 2.8 < len(wav) / sr < 3.3


def test_stereo_is_downmixed_to_mono(ffmpeg, source_wav, tmp_path):
    path = transcode(ffmpeg, source_wav, tmp_path / "stereo.m4a",
                     "-c:a", "aac", "-ac", "2")
    wav, _ = load_audio(path)
    assert wav.ndim == 1


def test_video_audio_track_is_extracted(ffmpeg, source_wav, tmp_path):
    path = tmp_path / "clip.mp4"
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "color=c=black:s=64x64:r=5",
         "-i", str(source_wav), "-shortest", "-c:a", "aac", str(path)],
        check=True, capture_output=True)
    wav, sr = load_audio(path)
    assert sr == SR
    assert len(wav) > SR


def test_a_missing_file_says_so():
    with pytest.raises(AudioLoadError, match="File not found"):
        load_audio("no_such_file.m4a")


def test_an_undecodable_file_reports_what_was_tried(tmp_path):
    """The failure this replaces surfaced as 'Could not decode x.m4a:' with
    nothing after the colon, because audioread's NoBackendError carries an
    empty message. Every decoder's own reason is now included."""
    path = tmp_path / "broken.m4a"
    path.write_bytes(b"not audio at all")
    with pytest.raises(AudioLoadError) as excinfo:
        load_audio(path)
    message = str(excinfo.value)
    assert "soundfile:" in message
    assert "ffmpeg:" in message
