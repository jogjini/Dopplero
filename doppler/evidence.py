"""Tone map of a recording and the scoring of a predicted aircraft curve on it (the lead's evidence measure,
docs/LEAD_REVIEW.md §4; first in scripts/site_evidence.py and scripts/place_search.py, shared here by them and the
app's job). The map: tonal contrast of the 0.25 s / 1 s channels, +-2 rows of slack, max over 0.5 s, ROWS rows.
A curve's score at a clock offset: the slope-weighted mean brightness under it at every base row of the engine
band (any engine pitch), as a robust z over rows; its best."""
from concurrent.futures import ProcessPoolExecutor

import numba
import numpy as np
from scipy import ndimage

from . import audio_io, locate, locate_job
from . import spectrogram as sg
from .adsb import label as lab

F_LO, F_HI = 30.0, 1500.0   # engine-tone band searched (props' fundamentals and first harmonics)
DEC = 4                      # map frames pooled: 0.5 s steps
STEP_S = DEC * sg.HOP_S      # 0.5 s: the tone map's and the scans' step
ROWS = 800                   # tone-map rows kept: the engine band plus the Doppler swing of a fast approach


def _core(job):
    """One 15-min core of the tone map (locate_job's chunking; forward_locate's map: tonal contrast of the
    0.25 s / 1 s channels, +-2 rows of slack, max over STEP_S), and which of its frames are recorded audio."""
    raw, g0 = job
    x = np.memmap(raw, np.float32, mode="r")
    i0, i1 = int(g0 * audio_io.SR), int((g0 + locate_job.CHUNK_S + 2 * locate_job.MARGIN_S) * audio_io.SR)
    seg, valid = np.zeros(i1 - i0, np.float32), np.zeros(i1 - i0, bool)
    a, b = max(i0, 0), min(i1, len(x))
    seg[a - i0:b - i0] = x[a:b]
    valid[a - i0:b - i0] = np.isfinite(seg[a - i0:b - i0])  # NaN = a gap between files
    seg[~valid] = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        F = sg.features(seg)
    W = np.maximum(F[0], F[1])
    ok = locate_job._frame_valid(valid, W.shape[1])
    W[:, ~ok] = np.nan
    c0, n = int(round(locate_job.MARGIN_S / sg.HOP_S)), int(round(locate_job.CHUNK_S / sg.HOP_S))
    M = ndimage.maximum_filter1d(np.nan_to_num(sg.tonal_contrast(W)[:, c0:c0 + n], nan=0.0), 5, axis=0)[:ROWS]
    return M.reshape(ROWS, -1, DEC).max(axis=2).astype(np.float16), ok[c0:c0 + n].reshape(-1, DEC).all(axis=1)


def tone_map(raw, dur, workers):
    """(M (ROWS, T) float16, valid (T,)) of the decoded recording at raw, T = dur / STEP_S."""
    starts = np.arange(-locate_job.MARGIN_S, dur, locate_job.CHUNK_S)
    with ProcessPoolExecutor(min(workers, len(starts))) as ex:
        parts = list(ex.map(_core, [(str(raw), float(g0)) for g0 in starts]))
    T = int(dur / STEP_S)
    return np.concatenate([p[0] for p in parts], axis=1)[:, :T], np.concatenate([p[1] for p in parts])[:T]


def passes(tracks, site, hg, t0, dur, c):
    """Every testable flying pass within locate.RADIUS_M of site (forward_locate.score_place's rules), reduced to
    its curve on the tone map as forward_locate.curve_z does: frames j, rows above the base row db, weights w."""
    frames = sg.frame_times(sg.n_frames(int(dur * audio_io.SR)))
    out = []
    for tr in tracks:  # one track at a time: a pass holds two recording-long arrays until it is reduced
        for p in lab.predict_passes([tr], (*site, hg + 1.5), t0, dur, frames, c, hg, locate.RADIUS_M, reach_s=locate.REACH_S):
            if p["on_ground_at_cpa"] or p["speed_cpa_mps"] < locate.MIN_FLYING_MPS:
                continue
            ratio = np.where(np.abs(frames - p["t_cpa_s"]) > locate.HALF_WINDOW_S, np.nan, p["ratio"])
            if not lab.testable(ratio):
                continue
            r = ratio[DEC // 2::DEC][:int(dur / STEP_S)]  # the tone map's frames
            j = np.flatnonzero(np.isfinite(r))
            if len(j) < 20:
                continue
            db = sg.BPO * np.log2(r[j])
            slope = np.abs(np.gradient(db))
            out.append({"icao": p["icao"], "type": p["type"], "t_cpa_s": p["t_cpa_s"], "d_cpa_m": p["d_cpa_m"],
                        "speed_mps": p["speed_cpa_mps"], "height_m": p["height_cpa_m"] - hg, "j": j,
                        "db": np.rint(db).astype(np.int64), "w": (0.2 + slope / (slope.max() + 1e-9)).astype(np.float32)})
    return out


@numba.njit(parallel=True, cache=True)
def _under(Mp, vp, j, db, w, lo, hi, S):
    """sc[b - lo, s] = sum over curve points k of w[k] * Mp[b + db[k], j[k] + s], s = 0..2S (clock offset s - S):
    Mp is the tone map wrapped by S frames at both ends and j counts unwrapped frames. cover[s] = the weight of
    the curve's points that sit on recorded audio (vp) at that offset."""
    n = 2 * S + 1
    sc = np.zeros((hi - lo, n), np.float32)
    cover = np.zeros(n, np.float32)
    for k in range(len(j)):
        for s in range(n):
            cover[s] += w[k] * vp[j[k] + s]
    for r in numba.prange(hi - lo):
        for k in range(len(j)):
            row = lo + r + db[k]
            if 0 <= row < Mp.shape[0]:
                for s in range(n):
                    sc[r, s] += w[k] * Mp[row, j[k] + s]
    return sc, cover


def wrap(M, valid, S):
    """The tone map and its valid frames with S frames of the other end attached at each end."""
    return (np.concatenate([M[:, -S:], M, M[:, :S]], axis=1).astype(np.float32),
            np.concatenate([valid[-S:], valid, valid[:S]]).astype(np.float32))


def profile(Mp, vp, p, S):
    """z (2S+1,) of one pass's predicted curve at clock offsets -S..S frames: per offset, the weighted brightness
    under the curve at every base row of the engine band as a robust z over rows; its best (curve_z's
    statistic). NaN where under 70 % of the curve lies on recorded audio."""
    lo, hi = int(sg.freq_to_bin(F_LO)), int(sg.freq_to_bin(F_HI))
    sc, cover = _under(Mp, vp, p["j"], p["db"], p["w"], lo, hi, S)
    sc = np.ascontiguousarray(sc.T)  # medians over rows, one offset per line
    med = np.median(sc, axis=1)
    z = (sc.max(axis=1) - med) / (1.4826 * np.median(np.abs(sc - med[:, None]), axis=1) + 1e-9)
    z[cover < 0.7 * p["w"].sum()] = np.nan
    return z


@numba.njit(parallel=True, cache=True)
def _under_at(Mp, vp, j, db, w, lo, hi, offs):
    """site_evidence._under at the given offsets only (frames into the wrapped map, j + offs >= 0): the weighted
    brightness under the curve at every base row of the band, and the curve's recorded-audio cover, per offset."""
    n = len(offs)
    sc = np.zeros((hi - lo, n), np.float32)
    cover = np.zeros(n, np.float32)
    for k in range(len(j)):
        for s in range(n):
            cover[s] += w[k] * vp[j[k] + offs[s]]
    for r in numba.prange(hi - lo):
        for k in range(len(j)):
            row = lo + r + db[k]
            if 0 <= row < Mp.shape[0]:
                for s in range(n):
                    sc[r, s] += w[k] * Mp[row, j[k] + offs[s]]
    return sc, cover


def profile_at(Mp, vp, p, S, offs):
    """site_evidence.profile at offsets offs (frames, -S..S) only: the same statistic, without the full sweep."""
    lo, hi = int(sg.freq_to_bin(F_LO)), int(sg.freq_to_bin(F_HI))
    sc, cover = _under_at(Mp, vp, p["j"], p["db"], p["w"], lo, hi, (np.asarray(offs) + S).astype(np.int64))
    sc = np.ascontiguousarray(sc.T)
    med = np.median(sc, axis=1)
    z = (sc.max(axis=1) - med) / (1.4826 * np.median(np.abs(sc - med[:, None]), axis=1) + 1e-9)
    z[cover < 0.7 * p["w"].sum()] = np.nan
    return z

