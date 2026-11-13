"""Decode any audio/video file to 16 kHz mono float32 via ffmpeg; codec round-trips."""
import functools
import subprocess

import numpy as np
from scipy import signal

SR = 16000
DEMUXER = {"adts": "aac"}  # muxer name -> demuxer name where they differ


def _run(cmd, data=None):
    p = subprocess.run(cmd, input=data, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:6])}...: {p.stderr.decode(errors='replace')[-500:]}")
    return p.stdout


def load(path, sr=SR, offset=0.0, duration=None):
    """Mono float32 at sr. offset/duration in seconds (ffmpeg seeks, so it is cheap).
    path may also be the file's bytes, e.g. a member read from a zip."""
    data = path if isinstance(path, bytes) else None
    cmd = ["ffmpeg", "-v", "error", "-ss", str(offset), "-i", "-" if data else str(path)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    return np.frombuffer(_run(cmd, data), dtype=np.float32).copy()


def to_raw(path, out, sr=SR):
    """Decode like load() into a raw float32 file (read back with np.memmap): long recordings
    without holding them in memory."""
    _run(["ffmpeg", "-v", "error", "-y", "-ss", "0.0", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", str(out)])


def save(path, x, sr=SR):
    """Write any format ffmpeg infers from the extension (.flac, .wav, ...)."""
    _run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-",
          str(path)], np.asarray(x, dtype=np.float32).tobytes())


@functools.cache
def duration(path):
    out = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                str(path)])
    return float(out)


@functools.cache
def encoders():
    out = _run(["ffmpeg", "-hide_banner", "-encoders"]).decode()
    rows = [line.split() for line in out.splitlines()]
    return {r[1] for r in rows if len(r) > 1 and len(r[0]) == 6 and r[0][0] in "VAS"}


def codec_roundtrip(x, sr, encode_args, fmt):
    """Encode x with ffmpeg (e.g. ["-c:a", "libmp3lame", "-b:a", "64k"], fmt "mp3") and decode
    back to float32 at sr, aligned to the input (codecs add a few ms of delay)."""
    raw = np.asarray(x, dtype=np.float32).tobytes()
    enc = _run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-",
                *encode_args, "-f", fmt, "-"], raw)
    y = np.frombuffer(_run(["ffmpeg", "-v", "error", "-f", DEMUXER.get(fmt, fmt), "-i", "-", "-ac", "1",
                            "-ar", str(sr), "-f", "f32le", "-"], enc), dtype=np.float32)
    return align(y, x, sr)


def align(y, ref, sr, max_lag_s=0.25):
    """Shift y so it lines up with ref (cross-correlation on the first 30 s), same length."""
    n = min(len(ref), len(y), 30 * sr)
    m = int(max_lag_s * sr)
    xc = signal.correlate(y[:n], ref[:n], mode="full", method="fft")
    mid = n - 1
    lag = int(np.argmax(np.abs(xc[mid - m:mid + m + 1]))) - m
    y = y[lag:] if lag > 0 else np.concatenate([np.zeros(-lag, np.float32), y])
    out = np.zeros(len(ref), np.float32)
    out[:min(len(y), len(ref))] = y[:len(ref)]
    return out
