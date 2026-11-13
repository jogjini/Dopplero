"""Predict each pass's curve from ADS-B, find it in the audio, write labels (DESIGN.md §8).

For every aircraft segment near the mic we run the shared emission-time model, which gives
the *shape* of its Doppler curve, f(t) = f_source * ratio(t), but not f_source. On a
log-frequency axis a tone of any source frequency is the same shape shifted vertically, so
we slide the shape over every frequency row and over a range of clock offsets (recorder
clocks are only good to a few seconds) and score how bright the spectrogram is along it.
Rows that stand out are tonal components of that aircraft; each is then traced.
"""
import json
import re
import subprocess
from datetime import datetime

import numpy as np
from scipy import ndimage

from .. import physics
from .. import spectrogram as sg
from ..spectrogram import tonal_contrast
from .tracks import interpolate, segments, stretch_boxes

MATCH_Z = 5.0      # best (offset, row) must stand this far above the row distribution (robust z)
CANDIDATE_Z = 4.0  # below MATCH_Z but above chance level seen on real clips: needs a human look
# From scripts/calibrate_livermore.py: 40 real passes vs 80 impossible decoy shapes on 1-2 min
# clips; decoys max 3.7, 35/40 passes >= 5 (34 traced on the hand curve).
# shortcut: chance grows with recording length; re-check with decoys on long (Eclipse) audio
COMPONENT_Z = 4.0  # further components at the chosen offset
VISIBLE_DB = 3.0   # traced points dimmer than this (whitened dB) are dropped
# A prediction shorter or flatter than this fits any bright line (on Eclipse site 706, short
# far-away predictions scored as high as impossible decoys); such passes are 'untestable'.
# Every Livermore pass clears it (shortest 36 s, smallest drop 279 cents).
MIN_DURATION_S = 30.0
MIN_SPAN_CENTS = 200.0
# The CPA is a geometric reference only with this much track on both sides of it: at a track
# end the "closest point" is just where the data stops, and its range is still falling.
CPA_SUPPORT_S = 10.0


def recording_info(path):
    """Duration plus, when the file carries them (iPhone/QuickTime), start time and position."""
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:format_tags",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True)
    fmt = json.loads(out.stdout)["format"]
    tags = {k.lower(): v for k, v in fmt.get("tags", {}).items()}
    info = {"duration_s": float(fmt["duration"]), "start_unix": None, "lat": None, "lon": None,
            "alt_msl_m": None, "start_source": None}
    if "com.apple.quicktime.creationdate" in tags:
        info["start_unix"] = datetime.fromisoformat(tags["com.apple.quicktime.creationdate"]).timestamp()
        info["start_source"] = "quicktime creationdate (1 s resolution)"
    m = re.match(r"([+-][\d.]+)([+-][\d.]+)([+-][\d.]+)?", tags.get("com.apple.quicktime.location.iso6709", ""))
    if m:
        info["lat"], info["lon"] = float(m.group(1)), float(m.group(2))
        info["alt_msl_m"] = float(m.group(3)) if m.group(3) else None
    return info


def predict_passes(tracks, mic_geo, start_unix, duration, frames, c, ground_h, radius_m, reach_s=None):
    """Every aircraft segment that comes within radius_m while the recording runs, with its
    CPA (recording-relative) and the predicted received/emitted frequency ratio at frames.
    reach_s (locate; None = the labeller's whole-segment prediction): a segment is predicted only
    from reach_s before the first to reach_s after the last stretch that can come within radius_m
    (tracks.stretch_boxes), and not at all if none can: the ratio is then finite only there, and the
    segment-wide checks (monotonic arrival, CPA support, testable) see that window."""
    passes = []
    for tr in tracks:
        for idx in segments(tr):
            lo = max(tr.t[idx[0]], start_unix - radius_m / c - 60)
            hi = min(tr.t[idx[-1]], start_unix + duration + 60)
            if hi - lo < 2:
                continue
            if reach_s is not None:
                bx, s0, s1 = stretch_boxes(tr, idx, mic_geo, lo, hi)
                gap = np.hypot(np.maximum(0, np.maximum(bx[:, 0], -bx[:, 1])), np.maximum(0, np.maximum(bx[:, 2], -bx[:, 3])))
                near = gap <= radius_m
                if not near.any():
                    continue  # never within radius_m: no pass (horizontal distance <= slant range)
                d = (lo + 0.05) - lo  # np.arange's actual step (0.05 rounded at lo's magnitude): the same grid points
                k0 = int(np.floor(max(0.0, s0[near].min() - reach_s - lo) / d))
                k1 = int(np.ceil(min(hi - lo, s1[near].max() + reach_s - lo) / d))
                te = lo + d * np.arange(k0, k1)
                te = te[te < hi]
            else:
                te = np.arange(lo, hi, 0.05)
            pos = interpolate(tr, idx, te, mic_geo, ground_h)
            ok = np.all(np.isfinite(pos), axis=1)
            te, pos = te[ok] - start_unix, pos[ok]
            if len(te) < 20:
                continue
            arr, r = physics.arrival_times(te, pos, np.zeros(3), c)
            heard = (arr >= 0) & (arr <= duration)
            if not heard.any() or r[heard].min() > radius_m or np.any(np.diff(arr) <= 0):
                continue
            te_f = physics.emission_times(frames, te, arr)
            ratio = np.interp(te_f, te, 1 / np.gradient(arr, te))  # dt_e/dt_r: NaN outside
            cpa = physics.cpa(te, pos, np.zeros(3), c)
            i = int(np.argmin(r))
            passes.append({
                "icao": tr.icao, "type": tr.type, "reg": tr.reg,
                "t_cpa_s": cpa["t_cpa"], "d_cpa_m": cpa["d_cpa"], "speed_cpa_mps": cpa["speed"],
                "height_cpa_m": float(pos[i, 2]), "cpa_inside_track": bool(te[i] - te[0] >= CPA_SUPPORT_S and te[-1] - te[i] >= CPA_SUPPORT_S),
                "on_ground_at_cpa": bool(tr.alt_kind[idx][np.argmin(np.abs(tr.t[idx] - start_unix - te[i]))] == "ground"),
                "min_range_heard_m": float(r[heard].min()),
                "alt_kind_at_cpa": str(tr.alt_kind[idx][np.argmin(np.abs(tr.t[idx] - start_unix - te[i]))]),
                "ratio": ratio, "range_m": np.interp(te_f, te, r)})
    return sorted(passes, key=lambda p: p["min_range_heard_m"])


def _robust_z(x, axis):
    med = np.nanmedian(x, axis=axis, keepdims=True)
    mad = 1.4826 * np.nanmedian(np.abs(x - med), axis=axis, keepdims=True)
    return (x - med) / (mad + 1e-9)


def _shift(a, s):
    """a delayed by s frames (NaN-filled)."""
    out = np.full_like(a, np.nan)
    if abs(s) >= len(a):
        return out
    if s >= 0:
        out[s:] = a[:len(a) - s]
    else:
        out[:s] = a[-s:]
    return out


def _scores(W, db, weight, shifts):
    """Weighted mean of W along the predicted shape, for every start row and frame shift."""
    F, _ = W.shape
    rows0 = np.arange(F)
    full = weight.sum()
    out = np.full((len(shifts), F), np.nan)
    for k, s in enumerate(shifts):
        d, w = _shift(db, s), _shift(weight, s)
        cols = np.nonzero(np.isfinite(d) & (w > 0))[0]
        if len(cols) < 8 or w[cols].sum() < 0.7 * full:
            continue
        rows = rows0[:, None] + np.rint(d[cols]).astype(int)[None, :]
        inb = (rows >= 0) & (rows < F)
        vals = np.where(inb, W[np.clip(rows, 0, F - 1), cols], np.nan)
        wv = np.where(np.isfinite(vals), w[cols], 0.0)  # off the map or in a recording gap (NaN)
        enough = wv.sum(axis=1) > 0.5 * w[cols].sum()
        out[k] = np.where(enough, np.nansum(vals * wv, axis=1) / (wv.sum(axis=1) + 1e-9), np.nan)
    return out


def match(W, ratio, max_offset_s, tolerance_bins=2, range_m=None):
    """Best clock offset and the frequency rows whose brightness follows the predicted shape.

    Frames are weighted towards where the predicted curve moves: a flat stretch matches any
    steady tone (hum, HVAC) and carries no Doppler evidence. Each point of the shape may sit
    ±tolerance_bins off the prediction (ADS-B and the straight-air model are not exact).
    Offsets where the shifted prediction overlaps less than 70 % of the recording's share at
    zero offset are skipped: too few frames make noise look like a match. With range_m,
    frames also count in proportion to 1/range: the aircraft is loudest near CPA."""
    F = W.shape[0]
    db = sg.BPO * np.log2(ratio)
    slope = np.abs(np.gradient(np.nan_to_num(db)))
    weight = np.where(np.isfinite(db), 0.2 + slope / (np.nanmax(slope) + 1e-9), 0.0)
    if range_m is not None:
        weight = weight * np.nan_to_num(np.nanmin(range_m) / range_m)
    S = int(round(max_offset_s / sg.HOP_S))
    shifts = list(range(-S, S + 1))
    Wt = ndimage.maximum_filter1d(W, 2 * tolerance_bins + 1, axis=0)
    z = _robust_z(_scores(Wt, db, weight, shifts), axis=1)
    if not np.isfinite(z).any():
        return None
    k, row = np.unravel_index(np.nanargmax(z), z.shape)
    zbest = float(z[k, row])
    # the slack blurs timing: refine the offset on the exact map, same row, within ±1 s
    fine = [s for s in range(shifts[k] - 8, shifts[k] + 9) if -S <= s <= S]
    exact = _scores(W, db, weight, fine)[:, row]
    if np.isfinite(exact).any():
        k = shifts.index(fine[int(np.nanargmax(exact))])
    best = {"offset_s": float(shifts[k] * sg.HOP_S), "z": zbest, "shift": int(shifts[k])}
    zr = np.nan_to_num(z[k], nan=-np.inf)
    peaks = [b for b in range(1, F - 1) if zr[b] >= COMPONENT_Z and zr[b] >= zr[b - 1] and zr[b] >= zr[b + 1]]
    picked = [int(row)]  # the best row always counts, even if the refined offset lowered its z
    for b in sorted(peaks, key=lambda b: -zr[b]):  # strongest first, keep ones >= 3 bins apart
        if all(abs(b - p) >= 3 for p in picked):
            picked.append(b)
    best["rows"] = [(int(b), float(zr[b])) for b in picked]
    best["db"] = _shift(db, best["shift"])
    return best


def trace(W, row, db, half_width=3):
    """Follow the brightest pixel within ±half_width rows of the predicted curve; sub-bin peak
    by parabola. Returns recording-relative times, frequencies and brightness (dB)."""
    F, T = W.shape
    ts, fs, vs = [], [], []
    for t in np.nonzero(np.isfinite(db))[0]:
        c = row + int(round(db[t]))
        lo, hi = max(c - half_width, 1), min(c + half_width + 1, F - 1)
        if hi <= lo or not np.isfinite(W[lo - 1:hi + 1, t]).all():
            continue  # off the map or in a recording gap
        b = lo + int(np.argmax(W[lo:hi, t]))
        a, m, z = W[b - 1, t], W[b, t], W[b + 1, t]
        if m < VISIBLE_DB:
            continue
        den = a - 2 * m + z
        off = float(np.clip(0.5 * (a - z) / den, -0.5, 0.5)) if den < 0 else 0.0
        ts.append(t * sg.HOP_S)
        fs.append(float(sg.FMIN * 2 ** ((b + off) / sg.BPO)))
        vs.append(float(m))
    return {"t_s": ts, "f_hz": fs, "db": vs}


def testable(ratio):
    """Is the predicted curve long and bent enough to be told apart from other lines?"""
    ok = np.isfinite(ratio)
    return bool(ok.sum() * sg.HOP_S >= MIN_DURATION_S
                and 1200 * np.log2(ratio[ok].max() / ratio[ok].min()) >= MIN_SPAN_CENTS)


def label(W, passes, max_offset_s):
    """Attach a match (or a 'not visible' / 'untestable' verdict) to each predicted pass."""
    W = tonal_contrast(W)
    for p in passes:
        if not testable(p["ratio"]):
            p.update(status="untestable", match_z=None, clock_offset_s=None, components=[])
            continue
        m = match(W, p["ratio"], max_offset_s, range_m=p["range_m"])
        z = m["z"] if m else -np.inf
        p["status"] = "matched" if z >= MATCH_Z else "candidate" if z >= CANDIDATE_Z else "not_visible"
        p["match_z"] = m["z"] if m else None
        p["clock_offset_s"] = m["offset_s"] if m else None
        p["components"] = []
        if p["status"] != "not_visible":
            for row, z in m["rows"]:
                p["components"].append({"f_at_cpa_hz": float(sg.FREQS[row]), "z": z,
                                        "contour": trace(W, row, m["db"])})
            p["predicted_db_shift"] = m["db"]
    return passes


def resolve_conflicts(passes, tol_cents=30.0, min_overlap=0.3):
    """Two aircraft can't own the same curve. Where traced contours of two visible passes lie on
    top of each other (>= min_overlap of the shorter one's points within tol_cents at the same
    frames), the lower-scoring pass becomes 'ambiguous'. Run on real passes only (not decoys)."""
    def overlap(a, b):
        best = 0.0
        for ca in a["components"]:
            for cb in b["components"]:
                ta = dict(zip(np.rint(np.array(ca["contour"]["t_s"]) / sg.HOP_S).astype(int), ca["contour"]["f_hz"]))
                tb = dict(zip(np.rint(np.array(cb["contour"]["t_s"]) / sg.HOP_S).astype(int), cb["contour"]["f_hz"]))
                common = ta.keys() & tb.keys()
                if common:
                    hits = sum(abs(1200 * np.log2(ta[k] / tb[k])) < tol_cents for k in common)
                    best = max(best, hits / min(len(ta), len(tb)))
        return best

    vis = sorted((p for p in passes if p["components"]), key=lambda p: -p["match_z"])
    for i, p in enumerate(vis):
        for q in vis[:i]:
            if q["status"] != "ambiguous" and overlap(p, q) >= min_overlap:
                p["status"], p["ambiguous_with"] = "ambiguous", q["icao"]
                break
    return passes
