"""Front-end: multi-resolution STFT mapped to a shared log-frequency grid, plus whitening.

Every channel uses the same grid: frames every HOP_S seconds (frame j centred at j*HOP_S),
N_BINS log-spaced rows from FMIN to FMAX at BPO bins per octave (DESIGN.md §1, §3).
"""
import functools
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage, signal, sparse

from .audio_io import SR

FMIN, FMAX, BPO = 15.0, 7500.0, 96
N_BINS = int(np.floor(BPO * np.log2(FMAX / FMIN))) + 1
FREQS = FMIN * 2.0 ** (np.arange(N_BINS) / BPO)
HOP_S = 0.125
WINDOWS_S = (0.25, 1.0, 4.0)
LONG_DECIM = 8  # the 4 s window runs at 2 kHz; rows above ~950 Hz come from the 1 s channel


def n_frames(n_samples, fs=SR):
    return n_samples // int(round(HOP_S * fs)) + 1


def frame_times(n):
    return np.arange(n) * HOP_S


def freq_to_bin(f):
    return BPO * np.log2(np.asarray(f, dtype=float) / FMIN)


@functools.cache
def _log_matrix(n_fft, fs):
    """Linear-FFT power -> log-bin power. Where FFT bins are finer than log bins we sum them
    with triangular weights (the log bin acts as a filter); where they are coarser we
    interpolate. Rows above the valid band are left empty."""
    f_lin = np.arange(n_fft // 2 + 1) * fs / n_fft
    df = fs / n_fft
    rows, cols, vals = [], [], []
    for b, fb in enumerate(FREQS):
        if fb > 0.95 * fs / 2:
            break
        if df < fb * (2 ** (1 / BPO) - 1):
            w = 1 - np.abs(BPO * np.log2(np.maximum(f_lin, 1e-9) / fb))
            idx = np.nonzero(w > 0)[0]
            wv = w[idx]
        else:
            i = int(fb // df)
            idx, frac = np.array([i, i + 1]), fb / df - i
            wv = np.array([1 - frac, frac])
        rows += [b] * len(idx)
        cols += list(idx)
        vals += list(wv)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(N_BINS, len(f_lin)), dtype=np.float32)


def power(x, win_s=1.0, fs=SR):
    """Log-frequency power spectrogram (N_BINS, n_frames). Rows outside the band of this
    window are NaN."""
    T = n_frames(len(x), fs)
    if win_s >= 4.0:
        x, fs = signal.resample_poly(x, 1, LONG_DECIM).astype(np.float32), fs // LONG_DECIM
    W, hop = int(win_s * fs), int(round(HOP_S * fs))
    n_fft = 1 << (W - 1).bit_length()
    xp = np.pad(np.asarray(x, np.float32), (W // 2, W // 2 + hop * 2))
    frames = sliding_window_view(xp, W)[::hop][:T]
    win = signal.windows.hann(W, sym=False).astype(np.float32)
    M = _log_matrix(n_fft, fs)
    out = np.full((N_BINS, T), np.nan, np.float32)
    valid = np.asarray(M.sum(axis=1)).ravel() > 0
    for j in range(0, T, 256):  # chunked: a full 4 s STFT of a long file would not fit in RAM
        spec = np.abs(np.fft.rfft(frames[j:j + 256] * win, n_fft)) ** 2
        out[valid, j:j + 256] = (M @ spec.T.astype(np.float32))[valid]
    return out


def whiten_db(P, block=8, span_blocks=31):
    """dB relative to each row's running median over ~span_blocks*block frames (~30 s).
    Medians are taken per block first: same result in practice, far cheaper."""
    finite = np.isfinite(P)
    floor = np.nanmax(P) * 1e-12 + 1e-30 if finite.any() else 1e-30
    L = 10 * np.log10(np.where(finite, P, np.nan) + floor)
    F, T = L.shape
    nb = -(-T // block)
    Lp = np.pad(L, ((0, 0), (0, nb * block - T)), mode="edge").reshape(F, nb, block)
    with warnings.catch_warnings():  # all-NaN rows (outside a window's band) stay NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(Lp, axis=2)
    med = ndimage.median_filter(med, size=(1, span_blocks), mode="nearest")
    centres = (np.arange(nb) + 0.5) * block - 0.5
    t = np.arange(T)
    base = np.stack([np.interp(t, centres, row) for row in med]) if nb > 1 else med[:, :1]
    return (L - base).astype(np.float32)


def tonal_contrast(W, half_width=7):
    """Brightness relative to the median of neighbouring frequencies: keeps narrow tones,
    cancels broadband sound (a takeoff roar brightens every row at once)."""
    return W - ndimage.median_filter(W, size=(2 * half_width + 1, 1), mode="nearest")


def features(x, fs=SR, span_blocks=31):
    """(len(WINDOWS_S), N_BINS, n_frames) whitened dB, the detector's input."""
    chans = [whiten_db(power(x, w, fs), span_blocks=span_blocks) for w in WINDOWS_S]
    for c in chans:  # fill out-of-band rows of the long window from the 1 s channel
        bad = ~np.isfinite(c)
        c[bad] = chans[1][bad]
    return np.nan_to_num(np.stack(chans), nan=0.0)
