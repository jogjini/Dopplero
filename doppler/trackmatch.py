"""Place search by curve shape against real flight tracks (owner's request 2026-11-11, for hand-fitted curves).

No curve model and no speed or distance read off a curve: for every place of a grid over the box, each flight's
real track (turns, descents and speed changes included) predicts the pitch heard there over time; a curve's misfit
at a place and clock offset is the best RMS (cents, its pitch profiled out per tone) over the flights airborne at
its time. One clock offset is shared by all curves (+-SLACK_S each for clicking error); a place's score is the best
over clock offsets of the sum over curves of CAP - min((misfit / SIGMA)^2, CAP): every curve must fit a real flight,
and a curve no flight explains adds nothing. A coarse grid first, then a finer grid around its best places.

Geometry is exact on the WGS84 ellipsoid: each track piece is computed in a local frame at the plane's own position,
and each place is the ground point (lat, lon) at the microphone's height in that frame (one flat frame over a
continent put far places tens of km above their real ground). Large boxes are searched around places proposed by each
curve x each flight's closest approach, not on an even grid (a curve fits only within a few hundred metres).

Prototype: scripts/handcurve_locate.py (the video, 4 hand-fitted curves, clock +-2 s: best of 5,625 places 0.40 km
from the camera, two of the curves from turning, descending jets that the straight-pass searches can't use).
"""
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numba
import numpy as np

from . import memguard
from .adsb.tracks import from_enu, interpolate, segments, to_enu

LEAD_S = 120.0        # emission can precede the heard point by this much (sound travel from ~40 km)
HEAR_M = 15000.0      # a flight farther than this (horizontally) from a place during a curve can't explain it
SIGMA, CAP = 10.0, 9.0
MAX_POINTS = 80       # points per curve (thinned evenly in time)
STEP_S, SLACK_S = 1.0, 1.0
COARSE_PLACES = 8000  # the coarse grid's size (its spacing follows the box)
MIN_SPACING_M = 300.0
MAX_GRID_SPACING_M = 600.0  # an even grid coarser than this misses the place: larger boxes use proposals
U_STEP_M, PROPOSE_CENTS = 250.0, 40.0  # proposals: places along each closest-approach line kept below this misfit
FALL_WINDOW_S, FALL_STEP_S = 20.0, 2.0   # the closest moment sought around the curve's half-way-pitch time
# each curve's table: its best misfit against a straight, steady pass at every (speed, closest distance)
TABLE_SPEEDS = 20.0 * (330.0 / 20.0) ** np.linspace(0, 1, 80)
TABLE_DISTS = 100.0 * (19000.0 / 100.0) ** np.linspace(0, 1, 100)
SAMPLE_S = 5.0          # proposing: each flight's closest-approach line every this many seconds near the box
UNSTEADY_CENTS = 60.0   # a turning or accelerating plane: the table's lenient cut, then the exact check (the video's
# turning jets fit the straight-line table at 26 and 47 cents at their true distance)
UNSTEADY_STEP_M = 1000.0  # ... along its line every this many metres (not every U_STEP_M)
STEADY_DEG, STEADY_FRAC = 5.0, 0.05  # steady over a curve's span: heading and speed change below these (15 deg / 10 %
# lost the video's King Air, slowing 9 % over its pass, to the straight-line table)
FLAT_FRAC = 0.3         # a curve fitting this share of the table's (speed, distance) cells says too little to propose
CELL_M, BUCKET_S = 2000.0, 5.0
EXACT_POOL, N_CHECK = 3000, 50  # a known start: the table's best places scored exactly; the best N_CHECK of those checked
M_PER_DEG = 111320.0
PROPOSAL_SPACING_M = 400.0
PIECE_STEP_S = 1.0  # flight paths sampled this often (linear between: cm-level at these speeds)
SLICE_S = 900.0   # a long clock window is proposed in slices of this many seconds of clock offset, one per worker
SLICE_KEEP = 300  # the best places each slice keeps (a slice over a continent proposes millions)
KEEP_FLIGHTS = 4000  # a slice keeps its curves' flights for the later passes while at most this many in all (else reads again)
# a slice whose flights x seconds of clock exceed this is done in halves, one after the other (memory: its candidate rows
# grow with both; a dense day over a continent filled a 16 GB Mac at 8 workers); not below MIN_SLICE_S
SLICE_FLIGHT_S, MIN_SLICE_S = 3.0e6, 120.0
AGREE_BUCKET_S = 15.0  # the two-pass slices' agreement test, in clock (see _agreeing)
N_SLICE_CHECK = 200  # date only: the best places of all slices, by exact score, checked on a grid
SLICE_EXACT_S = 6.0  # ... each scored exactly with every curve within this of its proposed clock offset
MAX_GRID_CLOCK_S = 60.0   # a longer clock window uses proposals even in a small box
CELL_SHIFT_S = BUCKET_S + 3.0  # a proposed place is checked within this of its proposed clock offset
N_REFINE, REFINE_K = 40, 4  # the best coarse places refined on a grid REFINE_K times finer


def curve_points(d):
    """(t in recording seconds, f, tone label) of one curve: a hand-fitted curve's clicked and snapped points, else
    its traced contours (each contour its own tone), thinned to MAX_POINTS."""
    if d.get("points"):
        P = np.asarray(d["points"], float)
    else:
        P = np.array([(t, f, k) for k, q in enumerate(d["contours"]) for t, f in zip(q["t_s"], q["f_hz"])], float)
    P = P[np.argsort(P[:, 0], kind="stable")]
    if len(P) <= MAX_POINTS:
        return P
    # thinned evenly along the line (1 s ~ 10 cents), not in time: a steep fall lasts seconds but carries the most shape
    step = np.hypot(np.diff(P[:, 0]), np.diff(1200 * np.log2(P[:, 1])) / 10.0)
    along = np.concatenate([[0.0], np.cumsum(step)])
    return P[np.unique(np.searchsorted(along, np.linspace(0, along[-1], MAX_POINTS)).clip(0, len(P) - 1))]


def min_speed(P, c, wind_mps):
    """Slowest ground speed that could make this curve's pitch fall: a tone's clicked span is at most the full
    Doppler span (c + v) / (c - v); less the wind allowance (sound follows the air, ADS-B the ground)."""
    r = max(P[P[:, 2] == h, 1].max() / P[P[:, 2] == h, 1].min() for h in np.unique(P[:, 2]))
    return max(0.0, c * (r - 1) / (r + 1) - wind_mps)


@numba.njit(cache=True, fastmath=True)
def _index(te, tau, step):
    """(j, w): tau lies between te[j - 1] and te[j], a fraction w of the way (held at the ends, as np.interp); step > 0:
    te is evenly spaced by step (direct lookup), else searched."""
    L = te.shape[0]
    if tau <= te[0]:
        return 1, 0.0
    if tau >= te[L - 1]:
        return L - 1, 1.0
    if step > 0:
        j = min(max(int((tau - te[0]) / step) + 1, 1), L - 1)
    else:
        j = np.searchsorted(te, tau)
    return j, (tau - te[j - 1]) / (te[j] - te[j - 1])


@numba.njit(cache=True, fastmath=True)
def _rms_kernel(Pt, V, te, X, T, f, perm, starts, c, wx, wy, step, dts):
    """_rms compiled: one place at a time, one point at a time (the search's innermost work). The emission time starts
    from the previous point's sound travel time (points are close in time, so the travel time barely changes). Sound
    travels at c through air moving at (wx, wy) m/s east/north: with d = listener - source, its travel time D solves
    |d - w D| = c D, and the heard / emitted pitch is 1 / (1 + dD/dtau) (without wind: c / (c + range rate)). Points
    grouped by tone: perm[starts[k]:starts[k + 1]] are tone k's (each tone's median is profiled out, in place)."""
    M, n = X.shape[0], T.shape[0]
    a = c * c - wx * wx - wy * wy
    out = np.empty(M)
    res = np.empty(n)
    buf = np.empty(n)
    for m in range(M):
        x0, x1, x2 = X[m, 0], X[m, 1], X[m, 2]
        covered = True
        delay = 0.0
        for i in range(n):
            t_i = T[i] + dts[m]  # each place may have its own clock offset
            tau = t_i - delay
            for _ in range(4 if i == 0 else 2):  # emission time: tau + D(tau) = heard time
                j, w = _index(te, tau, step)
                d0 = x0 - (Pt[j - 1, 0] + w * (Pt[j, 0] - Pt[j - 1, 0]))
                d1 = x1 - (Pt[j - 1, 1] + w * (Pt[j, 1] - Pt[j - 1, 1]))
                d2 = x2 - (Pt[j - 1, 2] + w * (Pt[j, 2] - Pt[j - 1, 2]))
                dw = d0 * wx + d1 * wy
                tau = t_i - (-dw + np.sqrt(dw * dw + a * (d0 * d0 + d1 * d1 + d2 * d2))) / a
            delay = t_i - tau
            if tau < te[0] or tau > te[te.shape[0] - 1]:
                covered = False
                break
            j, w = _index(te, tau, step)  # one lookup for the position and the velocity
            d0 = x0 - (Pt[j - 1, 0] + w * (Pt[j, 0] - Pt[j - 1, 0]))
            d1 = x1 - (Pt[j - 1, 1] + w * (Pt[j, 1] - Pt[j - 1, 1]))
            d2 = x2 - (Pt[j - 1, 2] + w * (Pt[j, 2] - Pt[j - 1, 2]))
            v0 = V[j - 1, 0] + w * (V[j, 0] - V[j - 1, 0])
            v1 = V[j - 1, 1] + w * (V[j, 1] - V[j - 1, 1])
            v2 = V[j - 1, 2] + w * (V[j, 2] - V[j - 1, 2])
            dw = d0 * wx + d1 * wy
            D = (-dw + np.sqrt(dw * dw + a * (d0 * d0 + d1 * d1 + d2 * d2))) / a
            dD = (D * (wx * v0 + wy * v1) - (d0 * v0 + d1 * v1 + d2 * v2)) / (a * D + dw)
            res[i] = 1200.0 * np.log2(f[i] * (1.0 + dD))
        if not covered:
            out[m] = np.inf
            continue
        for k in range(starts.shape[0] - 1):  # each tone's median, from a scratch copy sorted in place
            a0, a1 = starts[k], starts[k + 1]
            cnt = a1 - a0
            for q in range(cnt):
                buf[q] = res[perm[a0 + q]]
            b = buf[:cnt]
            b.sort()
            med = b[cnt // 2] if cnt % 2 else 0.5 * (b[cnt // 2 - 1] + b[cnt // 2])
            for q in range(cnt):
                res[perm[a0 + q]] -= med
        acc = 0.0
        for i in range(n):
            acc += res[i] * res[i]
        out[m] = np.sqrt(acc / n)
    return out


def _rms(Pt, V, te, X, T, f, tones, air, dts=None):
    """RMS misfit (cents, pitch profiled out per tone) at places X (M, 1, 3) of a curve heard at true times T (n,)
    against one track piece (positions Pt and velocities V at emission times te, same frame as X); inf where the track
    doesn't cover it. air = (speed of sound, wind east, wind north), m/s; dts: a clock offset per place (added to T)."""
    c, wx, wy = air
    tone = np.zeros(len(T), np.int64)
    for k, m in enumerate(tones):
        tone[m] = k
    perm = np.argsort(tone, kind="stable")
    starts = np.searchsorted(tone[perm], np.arange(len(tones) + 1))
    step = float(te[1] - te[0]) if len(te) > 1 else 0.0
    if len(te) > 1 and abs((te[-1] - te[0]) - step * (len(te) - 1)) > 1e-6:
        step = 0.0  # unevenly spaced (a piece with a dropped sample): searched
    return _rms_kernel(np.ascontiguousarray(Pt, float), np.ascontiguousarray(V, float), np.ascontiguousarray(te, float),
                       np.ascontiguousarray(X[:, 0, :], float), np.ascontiguousarray(T, float), np.ascontiguousarray(f, float),
                       perm, starts.astype(np.int64), float(c), float(wx), float(wy), step,
                       np.zeros(len(X)) if dts is None else np.ascontiguousarray(dts, float))


def _pieces(tr, t_lo, t_hi, hm, ground_h):
    """(emission times, positions, velocities, frame) of each continuous piece of a track within [t_lo, t_hi], every
    PIECE_STEP_S, in a local frame (lat, lon, hm) under the plane's middle position."""
    for idx in segments(tr):
        lo, hi = max(tr.t[idx[0]], t_lo), min(tr.t[idx[-1]], t_hi)
        if hi - lo < 10:
            continue
        mid = (lo + hi) / 2
        frame = (float(np.interp(mid, tr.t[idx], tr.lat[idx])), float(np.interp(mid, tr.t[idx], tr.lon[idx])), hm)
        te = np.arange(lo, hi, PIECE_STEP_S)
        Pt = interpolate(tr, idx, te, frame, ground_h)
        ok = np.isfinite(Pt).all(axis=1)
        if ok.sum() >= 20:
            yield te[ok], Pt[ok], np.gradient(Pt[ok], te[ok], axis=0), frame


def _places(lat, lon, frame):
    """Ground places (lat, lon) at the microphone's height, in a piece's frame: (M, 3), exact (curvature included)."""
    return to_enu(lat, lon, np.full(len(lat), frame[2]), *frame)


def _chunk(job):
    """Misfit (places x clock offsets, cents; inf where no flight covers the curve) and the best flight's index, over
    some flights. base (optional): each place's own clock offset, the shifts then relative to it."""
    flights, P, start, LAT, LON, c, ground_h, hm, shifts = job[:9]
    base = job[9] if len(job) > 9 else np.zeros(len(LAT))
    T, f, g = start + P[:, 0], P[:, 1], P[:, 2]
    tones = [g == h for h in np.unique(g)]
    best = np.full((len(LAT), len(shifts)), np.inf, np.float32)
    who = np.full((len(LAT), len(shifts)), -1, np.int32)
    for j, tr in flights:
        for te, Pt, V, frame in _pieces(tr, T.min() + base.min() + shifts.min() - LEAD_S, T.max() + base.max() + shifts.max() + 1.0, hm, ground_h):
            # a cheap first cut in degrees, then the exact places near the track
            pad = (HEAR_M + 30000.0) / M_PER_DEG
            la_s = np.interp(te[::10], tr.t, tr.lat)
            rough = np.flatnonzero((LAT >= la_s.min() - pad) & (LAT <= la_s.max() + pad))
            if not len(rough):
                continue
            Xr = _places(LAT[rough], LON[rough], frame)
            s = Pt[::10, :2]
            close = np.min(np.hypot(Xr[:, None, 0] - s[None, :, 0], Xr[:, None, 1] - s[None, :, 1]), axis=1) <= HEAR_M
            near = rough[close]
            if not len(near):
                continue
            X = Xr[close][:, None, :]
            for k, sh in enumerate(shifts):
                rms = _rms(Pt, V, te, X, T + sh, f, tones, c, base[near])  # when each point was really heard
                b = rms < best[near, k]
                best[near[b], k], who[near[b], k] = rms[b], j
    return best, who


def fall_time(P):
    """When (recording s) the pitch falls through halfway (log) between its first and last sixths, on the tone with the
    most points: around when the plane was heard closest (it comes a few seconds early)."""
    g = max(np.unique(P[:, 2]), key=lambda h: np.sum(P[:, 2] == h))
    Q = P[P[:, 2] == g]
    Q = Q[np.argsort(Q[:, 0])]
    y = np.log(Q[:, 1])
    k = max(1, len(Q) // 6)
    mid = (np.median(y[:k]) + np.median(y[-k:])) / 2
    i = int(np.argmax(y <= mid))
    if i == 0:
        return float(Q[len(Q) // 2, 0])
    w = (y[i - 1] - mid) / max(y[i - 1] - y[i], 1e-9)
    return float(Q[i - 1, 0] + w * (Q[i, 0] - Q[i - 1, 0]))


def curve_table(P, t_fall, c):
    """The curve against a straight, steady pass at every TABLE_SPEEDS x TABLE_DISTS: (best misfit, cents; the emission
    time of closest approach that gave it, relative to t_fall), over closest moments +-FALL_WINDOW_S. A straight line in
    3D stays one, so a climbing plane is covered by its 3D speed and the distance to its 3D line. No wind (the lookup
    allows a speed band; the exact check uses the real air)."""
    T, f, g = P[:, 0], P[:, 1], P[:, 2]
    tones = [g == h for h in np.unique(g)]
    dts = np.arange(-FALL_WINDOW_S, FALL_WINDOW_S + 1e-9, FALL_STEP_S)
    nd = len(TABLE_DISTS)
    # every (closest moment, distance) in one call per speed: a closest moment dt later is the same as hearing every
    # point dt earlier, so each place gets its own clock offset -dt against one track closest at t_fall
    X = np.tile(np.stack([np.zeros(nd), TABLE_DISTS, np.zeros(nd)], axis=1), (len(dts), 1))[:, None, :]
    shift = np.repeat(-dts, nd)
    te = np.arange(T.min() - FALL_WINDOW_S - 70.0, T.max() + FALL_WINDOW_S + 5.0, 0.5)
    best = np.full((len(TABLE_SPEEDS), nd), np.inf)
    when = np.zeros_like(best)
    for i, v in enumerate(TABLE_SPEEDS):
        V = np.tile([v, 0.0, 0.0], (len(te), 1))
        Pt = np.stack([v * (te - t_fall), np.zeros(len(te)), np.zeros(len(te))], axis=1)
        r = _rms(Pt, V, te, X, T, f, tones, (c, 0.0, 0.0), shift).reshape(len(dts), nd)
        k = np.argmin(r, axis=0)
        best[i], when[i] = r[k, np.arange(nd)], dts[k]
    return best, when


def _table_at(table, speed, dist, band):
    """The table's (misfit, emission offset) at a plane speed (allowing +- band m/s) and closest distances (arrays)."""
    best, when = table
    k = np.clip(np.rint(np.log(dist / TABLE_DISTS[0]) / np.log(TABLE_DISTS[-1] / TABLE_DISTS[0]) * (len(TABLE_DISTS) - 1)).astype(int),
                0, len(TABLE_DISTS) - 1)
    ok = (dist >= TABLE_DISTS[0]) & (dist <= TABLE_DISTS[-1])
    rows = np.flatnonzero((TABLE_SPEEDS >= speed - band) & (TABLE_SPEEDS <= speed + band))
    if not len(rows):  # a band narrower than the grid: the nearest speed
        rows = np.array([int(np.argmin(np.abs(TABLE_SPEEDS - speed)))])
    sub = best[rows][:, k]  # (speeds in the band, places)
    i = np.argmin(sub, axis=0)
    m = sub[i, np.arange(len(k))]
    m[~ok] = np.inf
    return m, when[rows[i], k]


def band_table(table, band):
    """The table with the speed band folded in: entry (speed i, distance k) = the best over speeds within +-band of
    speed i (and the emission offset that gave it), so a lookup is one array index."""
    best, when = table
    out_b, out_w = np.empty_like(best), np.empty_like(when)
    for i, v in enumerate(TABLE_SPEEDS):
        rows = np.flatnonzero(np.abs(TABLE_SPEEDS - v) <= max(band, 1e-9))
        rows = rows if len(rows) else np.array([i])
        j = np.argmin(best[rows], axis=0)
        out_b[i], out_w[i] = best[rows[j], np.arange(best.shape[1])], when[rows[j], np.arange(best.shape[1])]
    return out_b, out_w


KEEP_PER_SIDE = 2  # proposing keeps each plane-moment's best places on each side of its line (neighbours are redundant)


def _propose_chunk(job):
    """(n, 5) array (lat, lon, clock offset, misfit, flight) of the places that fit the curve along each flight's
    closest-approach lines. Every SAMPLE_S near the box: the line of ground places at right angles to the plane's
    velocity (turning or not), up to HEAR_M either side; each place's fit read from the curve's band table at the plane's
    speed and its distance, which also gives the clock offset (no scan); the best KEEP_PER_SIDE places on each side kept.
    A plane not flying steadily over the curve's span gets the table's lenient cut, then the exact calculation on the
    places kept. All of a track piece's moments at once (arrays), not one by one."""
    flights, P, t_fall, table, start, air, ground_h, hm, shifts, box, band = job[:11]
    defer = len(job) > 11 and job[11]  # unsteady planes: their candidates returned unchecked, flagged (column 5 = 1)
    btab, bwhen = table  # a band table (band_table)
    # per table speed, the largest distance at which the curve can fit at all (lenient cut): a plane-moment at a speed
    # with none, or higher above the listener than that, can't be the curve's plane there (high cruise traffic)
    fits = btab < UNSTEADY_CENTS
    reach = np.where(fits.any(axis=1), TABLE_DISTS[np.where(fits.any(axis=1), len(TABLE_DISTS) - 1 - np.argmax(fits[:, ::-1], axis=1), 0)], -1.0)
    T, f, g = start + P[:, 0], P[:, 1], P[:, 2]
    tones = [g == h for h in np.unique(g)]
    span = P[:, 0].max() - P[:, 0].min()
    u = np.arange(-HEAR_M, HEAR_M + 1.0, U_STEP_M)
    halves = (np.flatnonzero(u < 0), np.flatnonzero(u >= 0))
    pad = HEAR_M / M_PER_DEG
    ls, ld = np.log(TABLE_SPEEDS[-1] / TABLE_SPEEDS[0]), np.log(TABLE_DISTS[-1] / TABLE_DISTS[0])
    out = []
    at = lambda A, te, tau: np.stack([np.interp(tau, te, A[:, k]) for k in range(3)], axis=-1)
    for j, tr in flights:
        lo_t = start + t_fall + shifts.min() - FALL_WINDOW_S - 60.0
        hi_t = start + t_fall + shifts.max() + FALL_WINDOW_S
        for te, Pt, V, frame in _pieces(tr, lo_t - span - LEAD_S, hi_t + span + 5.0, hm, ground_h):
            # emission moments of closest approach to try: near the box, within the clock window, moving
            tc = np.arange(max(te[0] + span / 2, lo_t), min(te[-1] - span / 2, hi_t), SAMPLE_S)
            if not len(tc):
                continue
            la, lo = np.interp(tc, tr.t, tr.lat), np.interp(tc, tr.t, tr.lon)
            cosl = np.cos(np.radians(la))
            near = (la >= box[0] - pad) & (la <= box[1] + pad) & (lo >= box[2] - pad / cosl) & (lo <= box[3] + pad / cosl)
            Pe, Ve = at(Pt, te, tc[near]), at(V, te, tc[near])
            vh2 = Ve[:, 0] ** 2 + Ve[:, 1] ** 2
            i_sp = np.clip(np.rint(np.log(np.maximum(np.linalg.norm(Ve, axis=1), 1.0) / TABLE_SPEEDS[0]) / ls * (len(TABLE_SPEEDS) - 1)).astype(int),
                           0, len(TABLE_SPEEDS) - 1)
            mv = (vh2 >= 25.0) & (np.abs(Pe[:, 2]) <= reach[i_sp])  # moving, at a speed and height the curve allows
            tc, Pe, Ve, vh2 = tc[near][mv], Pe[mv], Ve[mv], vh2[mv]
            if not len(tc):
                continue
            # steady over the curve's span: heading and speed barely change
            Va, Vb = at(V, te, tc - span / 2), at(V, te, tc + span / 2)
            turn = np.degrees(np.abs(np.angle(np.exp(1j * (np.arctan2(Vb[:, 1], Vb[:, 0]) - np.arctan2(Va[:, 1], Va[:, 0]))))))
            sp = np.linalg.norm(Ve, axis=1)
            steady = (turn < STEADY_DEG) & (np.abs(np.linalg.norm(Vb, axis=1) - np.linalg.norm(Va, axis=1)) < STEADY_FRAC * np.maximum(sp, 1.0))
            # the ground lines (moments x places): (P - x) . V = 0, moved along the velocity when the plane climbs
            perp = np.stack([-Ve[:, 1], Ve[:, 0]], 1) / np.sqrt(vh2)[:, None]
            base = Pe[:, :2] + (Ve[:, 2] * Pe[:, 2] / vh2)[:, None] * Ve[:, :2]
            xh = base[:, None, :] + u[None, :, None] * perp[:, None, :]
            dist = np.sqrt(np.sum((xh - Pe[:, None, :2]) ** 2, axis=2) + Pe[:, 2:3] ** 2)
            i_s = np.clip(np.rint(np.log(sp / TABLE_SPEEDS[0]) / ls * (len(TABLE_SPEEDS) - 1)).astype(int), 0, len(TABLE_SPEEDS) - 1)
            k_d = np.clip(np.rint(np.log(np.maximum(dist, 1.0) / TABLE_DISTS[0]) / ld * (len(TABLE_DISTS) - 1)).astype(int), 0, len(TABLE_DISTS) - 1)
            m = np.where((dist >= TABLE_DISTS[0]) & (dist <= TABLE_DISTS[-1]), btab[i_s[:, None], k_d], np.inf)
            dt = bwhen[i_s[:, None], k_d]
            # a plane not flying steadily: the table assumed a straight, steady pass, so its ranking along the line can't
            # be trusted (a turning jet's real place can look worse than wrong ones); every place under the lenient cut
            # gets the exact calculation first, and the best few per side are chosen on that
            every = max(1, int(round(UNSTEADY_STEP_M / U_STEP_M)))  # the exact check along a coarser line (the check refines)
            uq, uu = np.nonzero(~steady[:, None] & (m < UNSTEADY_CENTS) & (np.arange(m.shape[1]) % every == 0)[None, :])
            if defer and len(uq):  # the candidates as they are; the exact check comes only where the other curves agree
                shu = tc[uq] - (start + t_fall + dt[uq, uu])
                ok = (shu >= shifts.min()) & (shu <= shifts.max())
                if ok.any():
                    xk = xh[uq[ok], uu[ok]]
                    lat, lon, _ = from_enu(xk[:, 0], xk[:, 1], np.zeros(len(xk)), *frame)
                    inb = (lat >= box[0]) & (lat <= box[1]) & (lon >= box[2]) & (lon <= box[3])
                    out.append(np.stack([lat[inb], lon[inb], shu[ok][inb], m[uq[ok], uu[ok]][inb], np.full(inb.sum(), j, float),
                                         np.ones(inb.sum())], axis=1))
                m[~steady] = np.inf  # the steady planes go on below
            elif len(uq):
                X = np.concatenate([xh[uq, uu], np.zeros((len(uq), 1))], axis=1)[:, None, :]
                m[uq, uu] = _rms(Pt, V, te, X, T, f, tones, air, tc[uq] - (start + t_fall + dt[uq, uu]))
                exact = np.zeros(m.shape, bool)
                exact[uq, uu] = True
                m[~steady[:, None] & ~exact] = np.inf  # the table's values aren't comparable with exact ones
            rows_q, rows_u = [], []  # (moment, place) pairs kept: each side's best few
            for h in halves:
                order = np.argsort(m[:, h], axis=1)[:, :KEEP_PER_SIDE]
                cols = h[order]
                ok = np.take_along_axis(m, cols, axis=1) < PROPOSE_CENTS
                qq = np.repeat(np.arange(len(tc))[:, None], cols.shape[1], axis=1)
                rows_q.append(qq[ok])
                rows_u.append(cols[ok])
            qi, ui = np.concatenate(rows_q), np.concatenate(rows_u)
            if not len(qi):
                continue
            sh = tc[qi] - (start + t_fall + dt[qi, ui])  # the clock offset: real closest moment minus the curve's
            mk = m[qi, ui]
            ok = (sh >= shifts.min()) & (sh <= shifts.max())
            if not ok.any():
                continue
            xk = xh[qi[ok], ui[ok]]
            lat, lon, _ = from_enu(xk[:, 0], xk[:, 1], np.zeros(len(xk)), *frame)
            inb = (lat >= box[0]) & (lat <= box[1]) & (lon >= box[2]) & (lon <= box[3])
            if inb.any():
                cols = [lat[inb], lon[inb], sh[ok][inb], mk[ok][inb], np.full(inb.sum(), j, float)]
                out.append(np.stack(cols + ([np.zeros(inb.sum())] if defer else []), axis=1))
    return np.concatenate(out) if out else np.zeros((0, 6 if defer else 5))


def _per_cell(rows, dlat, dlon):
    """rows (lat, lon, clock offset, misfit, curve) -> (cell keys, rows): the best row per cell x 5 s bucket x curve."""
    key = np.stack([np.floor(rows[:, 0] / dlat), np.floor(rows[:, 1] / dlon), np.rint(rows[:, 2] / BUCKET_S), rows[:, 4]], 1).astype(np.int64)
    order = np.lexsort((rows[:, 3], key[:, 3], key[:, 2], key[:, 1], key[:, 0]))
    key, rows = key[order], rows[order]
    first = np.ones(len(key), bool)
    first[1:] = np.any(key[1:] != key[:-1], axis=1)
    return key[first], rows[first]


def _rank(rows, dlat, dlon):
    """rows (lat, lon, clock offset, misfit, curve) -> (one row per place cell and clock bucket, how many curves agree
    there, its score), best score first. The score is the checking stage's: the sum over the curves fitted around the
    place of CAP - min((misfit / SIGMA)^2, CAP), each curve at its best misfit there. Counted per 2 km cell x 5 s bucket
    on four grids shifted by half a cell and half a bucket, each row taking its best: one place's proposals straddling a
    cell edge still meet. A place is put at the centre of its agreeing curves' best spots (each lies on one plane's
    closest-approach line; the place is where they cross). All in numpy: a continent's day gives millions of rows."""
    if not len(rows):
        return rows, np.zeros(0, int), np.zeros(0)
    counts, scores = np.zeros(len(rows), np.int64), np.full(len(rows), -1.0)
    centre = rows[:, :3].copy()
    om = np.argsort(rows[:, 3], kind="stable")  # by misfit: a (place, curve)'s first row is its best
    curve = rows[:, 4].astype(np.int64)
    for o, ob in ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)):
        place = _pack(np.floor(rows[:, 0] / dlat + o).astype(np.int64), np.floor(rows[:, 1] / dlon + o).astype(np.int64),
                      np.floor(rows[:, 2] / BUCKET_S + ob).astype(np.int64))
        ukc, first, inv_kc = np.unique((place * 64 + curve)[om], return_index=True, return_inverse=True)
        _, inv_k = np.unique(ukc // 64, return_inverse=True)
        bestrow = rows[om[first]]  # each (place, curve)'s best row
        part = CAP - np.minimum((bestrow[:, 3] / SIGMA) ** 2, CAP)
        n_k, s_k = np.bincount(inv_k), np.bincount(inv_k, weights=part)
        c_k = np.stack([np.bincount(inv_k, weights=bestrow[:, i]) for i in range(3)], 1) / n_k[:, None]
        back = np.empty(len(rows), np.int64)
        back[om] = inv_k[inv_kc.ravel()]
        better = s_k[back] > scores
        counts, scores = np.where(better, n_k[back], counts), np.where(better, s_k[back], scores)
        centre[better] = c_k[back[better]]
    out = rows.copy()
    out[:, :3] = centre
    base = _pack(np.floor(out[:, 0] / dlat).astype(np.int64), np.floor(out[:, 1] / dlon).astype(np.int64),
                 np.rint(out[:, 2] / BUCKET_S).astype(np.int64))
    order = np.lexsort((out[:, 3], -scores))
    _, firsts = np.unique(base[order], return_index=True)
    sel = order[np.sort(firsts)]
    return out[sel], counts[sel], scores[sel]


def _exact_scores(curves, loader, start, air, ground_h, hm, LAT, LON, base, kept=None, ex=None, workers=1):
    """(score, clock offset) of each place at its own clock offset +- SLICE_EXACT_S: every curve against every flight's
    real track there (the checking stage's score, one shared clock, +- SLACK_S per curve). In this process, or with ex
    each curve's flights split over that pool's workers. kept: flights already read per curve (by its index), reused
    instead of read again."""
    rel = np.arange(-SLICE_EXACT_S, SLICE_EXACT_S + 1e-9, STEP_S)
    total = np.zeros((len(LAT), len(rel)))
    slack = int(round(SLACK_S / STEP_S))
    for k, (P, v_min) in enumerate(curves):
        flights = (kept or {}).get(k)
        if flights is None:
            flights = loader(start + P[:, 0].min() + base.min() + rel.min() - LEAD_S - 120.0, start + P[:, 0].max() + base.max() + rel.max() + 60.0)
        lo, hi = start + P[:, 0].min() + base.min() + rel.min() - LEAD_S, start + P[:, 0].max() + base.max() + rel.max()
        cand = [(j, tr) for j, tr in enumerate(flights) if tr.t[0] <= hi and tr.t[-1] >= lo and np.nanmax(np.nan_to_num(tr.gs)) >= v_min]
        n = max(1, min(len(cand), workers)) if ex is not None else 1
        parts = [(cand[i::n], P, start, LAT, LON, air, ground_h, hm, rel, base) for i in range(n)]
        best = np.min([b for b, _ in (ex.map(_chunk, parts) if ex is not None else map(_chunk, parts))], axis=0)
        for dk in range(1, slack + 1):  # each curve may sit a little off the shared clock
            for sgn in (1, -1):
                b2 = np.roll(best, sgn * dk, axis=1)
                (b2[:, :dk] if sgn > 0 else b2[:, -dk:])[:] = np.inf
                best = np.minimum(best, b2)
        total += CAP - np.minimum((best / SIGMA) ** 2, CAP)
        del flights, cand
    k = total.argmax(axis=1)
    return total[np.arange(len(LAT)), k], base + rel[k]


def _pack(*cols):
    """Integer columns -> one int64 key per row (np.unique on it is a plain sort, far faster than on rows)."""
    lo = [c.min() for c in cols]
    dims = tuple(int(c.max() - m) + 1 for c, m in zip(cols, lo))
    return np.ravel_multi_index(tuple(c - m for c, m in zip(cols, lo)), dims)


def _agreeing(rows, dlat, dlon, need, bucket_s=AGREE_BUCKET_S):
    """Which rows (lat, lon, clock offset, misfit, curve) lie where at least need curves have a row (2 km x bucket_s, on
    the four half-shifted grids). Coarser in clock than the ranking: a turning plane's clock offset comes from the
    straight-line table and can be seconds off before its exact check."""
    hot = np.zeros(len(rows), bool)
    if not len(rows):
        return hot
    curve = rows[:, 4].astype(np.int64)
    for o, ob in ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)):
        a, b, t = (np.floor(rows[:, 0] / dlat + o).astype(np.int64), np.floor(rows[:, 1] / dlon + o).astype(np.int64),
                   np.floor(rows[:, 2] / bucket_s + ob).astype(np.int64))
        place = _pack(a, b, t)
        ukc, inv_kc = np.unique(place * 64 + curve, return_inverse=True)  # (place, curve) pairs; curves < 64
        _, inv_k = np.unique(ukc // 64, return_inverse=True)
        hot |= np.bincount(inv_k)[inv_k][inv_kc.ravel()] >= need
    return hot


def _exact_rows(rows, flights, P, t_fall, start, air, ground_h, hm, shifts):
    """The exact misfit of candidate rows (lat, lon, clock offset, ., flight) of one curve: each against its own
    flight's real track at its own clock offset."""
    T, f, g = start + P[:, 0], P[:, 1], P[:, 2]
    tones = [g == h for h in np.unique(g)]
    span = P[:, 0].max() - P[:, 0].min()
    lo = start + t_fall + shifts.min() - FALL_WINDOW_S - 60.0 - span - LEAD_S
    hi = start + t_fall + shifts.max() + FALL_WINDOW_S + span + 5.0
    out = np.full(len(rows), np.inf)
    for j in np.unique(rows[:, 4]).astype(int):
        idx = np.flatnonzero(rows[:, 4] == j)
        for te, Pt, V, frame in _pieces(flights[j], lo, hi, hm, ground_h):
            X = _places(rows[idx, 0], rows[idx, 1], frame)[:, None, :]
            out[idx] = np.minimum(out[idx], _rms(Pt, V, te, X, T, f, tones, air, rows[idx, 2]))
    return out


def _slice_job(job):
    """One slice of clock offsets in a worker, flights read here (the parent holds none), each curve with the flights of
    its own moment only. Pass 1: every curve's table proposals; turning or slowing planes only flagged as candidates.
    Pass 2: those candidates checked exactly only where all the proposing curves but one have a proposal (elsewhere the
    place can't win), then the places ranked, and the best SLICE_KEEP scored exactly with every curve at their own
    clock (a day over a continent has many places where most curves fit the
    straight-line table by chance; the exact score of all curves at one clock tells them apart). Saved to path (a
    slice already saved is not redone)."""
    path = job[9]
    if path.exists():
        return path
    out = _slice_places(job)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, out)
    tmp.replace(path)  # only a finished slice counts as done
    return path


def _slice_places(job):
    """A slice's best places (see _slice_job); a dense one in halves of its clock offsets."""
    loader, items, start, air, ground_h, hm, shifts, box, band, path, dlat, dlon, every = job
    window = lambda P: (start + P[:, 0].min() + shifts.min() - FALL_WINDOW_S - LEAD_S - 120.0,
                        start + P[:, 0].max() + shifts.max() + FALL_WINDOW_S + 60.0)
    rows, kept = [], {}  # kept: the curves' flights reused by the later passes when few (a small box), else read again
    for k, P, t_fall, table, v_min in items:
        flights = loader(*window(P))
        if not rows and not kept and len(flights) * np.ptp(shifts) > SLICE_FLIGHT_S and np.ptp(shifts) >= 2 * MIN_SLICE_S:
            del flights
            mid = shifts.min() + np.ptp(shifts) / 2
            halves = [_slice_places(job[:6] + (h,) + job[7:]) for h in (shifts[shifts < mid + BUCKET_S], shifts[shifts >= mid - BUCKET_S])]
            out = np.concatenate(halves)  # each half's best, scored exactly: comparable
            return out[np.argsort(-out[:, 6], kind="stable")][:SLICE_KEEP]
        t = start + t_fall
        cand = [(j, tr) for j, tr in enumerate(flights) if tr.t[0] <= t + shifts.max() + FALL_WINDOW_S
                and tr.t[-1] >= t + shifts.min() - FALL_WINDOW_S - LEAD_S and np.nanmax(np.nan_to_num(tr.gs)) >= v_min]
        r = _propose_chunk((cand, P, t_fall, table, start, air, ground_h, hm, shifts, box, band, True))
        if len(flights) + sum(map(len, kept.values())) <= KEEP_FLIGHTS:
            kept[k] = flights
        del flights, cand
        if len(r):
            rows.append(np.concatenate([r, np.full((len(r), 1), k, float)], 1))  # lat lon clock misfit flight unsteady curve
    if rows:
        R = np.concatenate(rows)
        need = max(1, len(items) - 1)
        hot = _agreeing(R[:, [0, 1, 2, 3, 6]], dlat, dlon, need)
        check = (R[:, 5] == 1) & hot
        for k, P, t_fall, table, v_min in items:
            sel = np.flatnonzero(check & (R[:, 6] == k))
            if len(sel):
                R[sel, 3] = _exact_rows(R[sel], kept[k] if k in kept else loader(*window(P)), P, t_fall, start, air, ground_h, hm, shifts)
        R = R[(R[:, 5] == 0) | (check & (R[:, 3] < PROPOSE_CENTS))]
        rows = [_per_cell(R[:, [0, 1, 2, 3, 6]], dlat, dlon)[1]] if len(R) else []
    ranked, counts, scores = _rank(np.concatenate(rows), dlat, dlon) if rows else (np.zeros((0, 5)), np.zeros(0, int), np.zeros(0))
    out = np.concatenate([ranked, counts[:, None].astype(float), scores[:, None]], 1)[:SLICE_KEEP]
    if len(out):  # the exact score, every curve at one clock, replaces the table's
        out[:, 6], out[:, 2] = _exact_scores(every, loader, start, air, ground_h, hm, out[:, 0], out[:, 1], out[:, 2], kept)
        out = out[np.argsort(-out[:, 6], kind="stable")]
    return out


def _propose(curves, flights, start, c, ground_h, hm, shifts, wind_mps, box, ex, workers, say, loader=None, work_dir=None, info=None):
    """Candidate places [(lat, lon, clock offset)] for a large box or a long clock window: the places where curves fit
    some flight around them (2 km, 5 s), ranked by the table's score; the best EXACT_POOL scored exactly (every curve,
    one clock) and the best N_CHECK of those kept (date only: the slices' best, N_SLICE_CHECK). Every curve proposes, through its table, except those whose table
    fits almost anything (a nearly flat tone says little about the place). With a loader (a long window, e.g. the date
    only), the clock window is cut into SLICE_S slices, each a worker that reads its own flights and saves its best
    places to work_dir."""
    dlat = CELL_M / M_PER_DEG
    dlon = dlat / np.cos(np.radians((box[0] + box[1]) / 2))
    band = min(float(wind_mps), 15.0)  # the table's speed band: ground speed vs speed through the air
    raw = [curve_table(P, fall_time(P), c[0]) for P in curves]
    anchors = [k for k, (b, _) in enumerate(raw) if np.mean(b < PROPOSE_CENTS) < FLAT_FRAC]
    tables = [band_table(t, band) for t in raw]
    if not anchors:
        anchors = list(range(len(curves)))
    say(f"proposing: {len(anchors)} of {len(curves)} curves propose (the others' tones are too flat)")
    if loader is not None:
        work = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="trackmatch_"))
        work.mkdir(parents=True, exist_ok=True)
        items = [(k, curves[k], fall_time(curves[k]), tables[k], min_speed(curves[k], c[0], wind_mps)) for k in anchors]
        edges = np.arange(shifts.min(), shifts.max() + 1e-9, SLICE_S)
        every = [(P, min_speed(P, c[0], wind_mps)) for P in curves]  # all curves score, not only the proposing ones
        jobs = [(loader, items, start, c, ground_h, hm, shifts[(shifts >= e - BUCKET_S) & (shifts < e + SLICE_S + BUCKET_S)], box, band,
                 work / f"slice_{i:03d}.npy", dlat, dlon, every) for i, e in enumerate(edges)]  # slices overlap by a bucket
        n = max(1, workers)
        while todo := [j for j in jobs if not j[9].exists()]:  # a slice saved is done (also after a memory stop)
            hit = []
            try:
                with ProcessPoolExecutor(n) as pool, memguard.guard(pool) as hit:
                    for _ in pool.map(_slice_job, todo):
                        b = sum(j[9].exists() for j in jobs)
                        say(f"proposing places: {round(100 * b / len(jobs))}% (slice {b} of {len(jobs)}, {SLICE_S / 60:.0f} min of start times each)")
            except BrokenProcessPool:
                if not hit:  # a worker died of something else
                    raise
                if n == 1:
                    raise MemoryError("memory nearly full even with one worker: close other apps, or search a smaller box") from None
                n //= 2
                time.sleep(3)  # the killed workers' memory comes back
                say(f"memory nearly full: the workers were stopped; going on with {n} (the slices already done are kept)")
        kept = [np.load(j[9]) for j in jobs]
        best = np.concatenate([k for k in kept if k.shape[1:] == (7,)]) if kept else np.zeros((0, 7))
        # the slices' best places, ranked once more (the overlap between slices gives some twice)
        base = np.stack([np.floor(best[:, 0] / dlat), np.floor(best[:, 1] / dlon), np.rint(best[:, 2] / BUCKET_S)], 1).astype(np.int64)
        order = np.lexsort((best[:, 3], -best[:, 6]))
        _, first = np.unique(base[order], axis=0, return_index=True) if len(best) else (None, np.zeros(0, int))
        sel = order[np.sort(first)]
        if info is not None:
            info["slice_best"] = best[:, [0, 1, 2, 6]]  # every slice's best places: (lat, lon, clock offset, exact score)
        # ranked by the exact score alone: the proposal count misses turning planes (Eclipse 172: the true place's best
        # rows had 3-4 of 6 proposals and were dropped by the count filter)
        say(f"proposing: {len(sel)} places ranked by their exact score; the best {N_SLICE_CHECK} checked")
        return [(r[0], r[1], r[2]) for r in best[sel[:N_SLICE_CHECK]]]
    else:
        rows = []
        for a, k in enumerate(anchors):
            P = curves[k]
            t_fall = fall_time(P)
            v_min = min_speed(P, c[0], wind_mps)
            t = start + t_fall
            cand = [(j, tr) for j, tr in enumerate(flights) if tr.t[0] <= t + shifts.max() + FALL_WINDOW_S
                    and tr.t[-1] >= t + shifts.min() - FALL_WINDOW_S - LEAD_S and np.nanmax(np.nan_to_num(tr.gs)) >= v_min]
            n = max(1, min(len(cand), 4 * workers))  # 4 batches per worker: progress lines and even loads
            jobs = [(cand[i::n], P, t_fall, tables[k], start, c, ground_h, hm, shifts, box, band) for i in range(n)]
            for b, r in enumerate(ex.map(_propose_chunk, jobs)):
                say(f"proposing places: {round(100 * (a + (b + 1) / n) / len(anchors))}% (curve {k + 1}, {a + 1} of {len(anchors)})")
                if len(r):
                    r[:, 4] = k
                    rows.append(_per_cell(r, dlat, dlon)[1])
        ranked, counts, _ = _rank(np.concatenate(rows), dlat, dlon) if rows else (np.zeros((0, 5)), np.zeros(0, int), np.zeros(0))
    if not len(ranked):
        say("proposing: no place fits")
        return []
    # the table's best, ranked again by the exact score: a count filter (all agreeing curves but one) let chance win over
    # a continent (Eclipse 724, ~20 000 aircraft: a wrong place had 7 of 9 curves agree, the true one fewer)
    pool = ranked[:EXACT_POOL]
    say(f"proposing: {len(ranked)} places ranked by the table; scoring the best {len(pool)} exactly")
    every = [(P, min_speed(P, c[0], wind_mps)) for P in curves]
    exact, clock = _exact_scores(every, lambda *_: flights, start, c, ground_h, hm, pool[:, 0], pool[:, 1], pool[:, 2], ex=ex, workers=workers)
    order = np.argsort(-exact, kind="stable")[:N_CHECK]  # the true place ranks near the top by exact score; each check costs ~6 s
    say(f"proposing: the best {len(order)} by exact score checked (most curves fitted around one place: {int(counts.max())} of {len(anchors)})")
    return [(pool[i, 0], pool[i, 1], clock[i]) for i in order]


def _score(curves, flights, start, LAT, LON, c, ground_h, hm, shifts, wind_mps, ex, workers, say, label):
    """(score per place and clock offset, per curve: misfit and best flight per place and clock offset)."""
    total = np.zeros((len(LAT), len(shifts)))
    per = []
    slack = int(round(SLACK_S / STEP_S))
    for n_done, P in enumerate(curves):
        T = start + P[:, 0]
        v_min = min_speed(P, c[0], wind_mps)
        cand = [(j, tr) for j, tr in enumerate(flights)
                if tr.t[0] <= T.max() + shifts.max() and tr.t[-1] >= T.min() + shifts.min() - LEAD_S and np.nanmax(np.nan_to_num(tr.gs)) >= v_min]
        n = max(1, min(len(cand), workers))
        jobs = [(cand[i::n], P, start, LAT, LON, c, ground_h, hm, shifts) for i in range(n)]
        best = np.full((len(LAT), len(shifts)), np.inf, np.float32)
        who = np.full((len(LAT), len(shifts)), -1, np.int32)
        for b, w in ex.map(_chunk, jobs):
            m = b < best
            best[m], who[m] = b[m], w[m]
        for dk in range(1, slack + 1):  # each curve may sit a little off the shared clock (clicking error)
            for sgn in (1, -1):
                b2, w2 = np.roll(best, sgn * dk, axis=1), np.roll(who, sgn * dk, axis=1)
                (b2[:, :dk] if sgn > 0 else b2[:, -dk:])[:] = np.inf
                m = b2 < best
                best, who = np.where(m, b2, best), np.where(m, w2, who)
        total += CAP - np.minimum((best / SIGMA) ** 2, CAP)
        per.append((best, who))
        say(f"{label}: {round(100 * (n_done + 1) / len(curves))}% (curve {n_done + 1} of {len(curves)})")
    return total, per


def _grid(lat_c, lon_c, half_m, spacing, box=None):
    """Places (lat, lon) of a square grid spacing m apart, half_m either side of a centre (clipped to box)."""
    d = np.arange(-half_m, half_m + 1e-6, spacing) / M_PER_DEG
    la, lo = np.meshgrid(lat_c + d, lon_c + d / np.cos(np.radians(lat_c)))
    la, lo = la.ravel(), lo.ravel()
    if box is not None:
        m = (la >= box[0]) & (la <= box[1]) & (lo >= box[2]) & (lo <= box[3])
        la, lo = la[m], lo[m]
    return la, lo


def _ranked(curves, flights, start, LAT, LON, c, ground_h, hm, shifts, wind_mps, ex, workers, say, label):
    """[{lat, lon, score, clock_s, source, matches}] of every place given, at its best clock offset of shifts."""
    total, per = _score(curves, flights, start, LAT, LON, c, ground_h, hm, shifts, wind_mps, ex, workers, say, label)
    k = total.argmax(axis=1)
    out = []
    for i in range(len(LAT)):
        matches = [{"icao": flights[w].icao, "type": flights[w].type, "residual_cents": float(b), "curve": n}
                   for n, (best, who) in enumerate(per) if (w := int(who[i, k[i]])) >= 0 and np.isfinite(b := best[i, k[i]])]
        out.append({"lat": float(LAT[i]), "lon": float(LON[i]), "score": float(total[i, k[i]]), "clock_s": float(shifts[k[i]]),
                    "source": "tracks", "matches": matches})
    return out


CHANCE_CLOCK_S, CHANCE_KM = 600.0, 30.0  # "elsewhere": another start time this far off, or a place this far away


def _chance(slice_best, winner):
    """The chance level a date-only search gives for free: every slice scored its best places exactly (all curves at one
    clock), so the best of them at another start time (CHANCE_CLOCK_S off) or another place (CHANCE_KM away) is what
    chance reaches that day; the winner's own score at that same stage is its fair comparison (both before the fine
    grid)."""
    km = np.hypot((slice_best[:, 0] - winner["lat"]) * M_PER_DEG,
                  (slice_best[:, 1] - winner["lon"]) * M_PER_DEG * np.cos(np.radians(winner["lat"]))) / 1000
    far = (np.abs(slice_best[:, 2] - winner["clock_s"]) > CHANCE_CLOCK_S) | (km > CHANCE_KM)
    return {"chance_score": float(slice_best[far, 3].max()) if far.any() else None,
            "winner_same_stage": float(slice_best[~far, 3].max()) if (~far).any() else None}


def _by_clock(items, half_s, max_span_s=20.0):
    """[(lat, lon, clock offset)] -> [(items, clock offsets to check)]: items whose offsets lie within max_span_s of the
    group's first are checked together, over the group's offsets +- half_s (one scoring pass instead of one each)."""
    out = []
    for it in sorted(items, key=lambda q: q[2]):
        if out and it[2] - out[-1][0][0][2] <= max_span_s:
            out[-1][0].append(it)
        else:
            out.append(([it], None))
    return [(g, np.arange(round(g[0][2]) - half_s, round(g[-1][2]) + half_s + 1e-9, STEP_S)) for g, _ in out]


def search(dets, flights, start, box, frame, ground_h, c, clock_s, workers=4, wind_mps=25.0, top_n=20, say=print, wind=(0.0, 0.0),
           loader=None, work_dir=None, info=None):
    """Ranked places [{lat, lon, score, clock_s, source, matches: [{icao, type, residual_cents, curve}]}] for the
    curves of dets in the flights of one sky; clock_s: the points were really heard clock_s later than the recording's
    clock says. frame = (lat0, lon0, mic height above the ellipsoid). A small box with a short clock window: an even
    grid at every clock offset. Otherwise (a large box, or the date only: clock_s up to a day): places proposed by
    each curve x each flight's closest approach, each checked at its own clock offsets only. loader(t0, t1) -> flights,
    instead of flights (None): each stage loads only the flights it needs (a whole day held at once is gigabytes);
    work_dir keeps the slices proposed (see _propose). info (a dict): with a loader, filled with the chance level (see
    _chance)."""
    curves = [curve_points(d) for d in dets if d.get("points") or d.get("contours")]
    curves = [P for P in curves if len(P) >= 6]
    if not curves:
        return []
    hm = frame[2]
    lat_c = (box[0] + box[1]) / 2
    area = (box[1] - box[0]) * M_PER_DEG * (box[3] - box[2]) * M_PER_DEG * np.cos(np.radians(lat_c))
    spacing = max(MIN_SPACING_M, float(np.sqrt(area / COARSE_PLACES)))
    shifts = np.arange(-clock_s, clock_s + 1e-9, STEP_S)
    air = (float(c), float(wind[0]), float(wind[1]))  # the speed of sound and the wind (east, north), m/s
    t_lo, t_hi = min(P[:, 0].min() for P in curves), max(P[:, 0].max() for P in curves)
    # the flights a check at these clock offsets needs: the given list, or loaded for just that window
    fl = lambda sh: flights if loader is None else loader(start + t_lo + sh[0] - LEAD_S - 120.0, start + t_hi + sh[-1] + 60.0)
    run = lambda LAT, LON, sh, lab, talk=say: _ranked(curves, fl(sh), start, LAT, LON, air, ground_h, hm, sh, wind_mps, ex, workers, talk, lab)
    stack = lambda g: np.unique(np.round(np.stack([np.concatenate([x[0] for x in g]), np.concatenate([x[1] for x in g])], 1), 7), axis=0).T
    with ProcessPoolExecutor(max(1, workers)) as ex:
        if spacing <= MAX_GRID_SPACING_M and clock_s <= MAX_GRID_CLOCK_S:  # every place of an even grid, every offset
            dlat = spacing / M_PER_DEG
            dlon = dlat / np.cos(np.radians(lat_c))
            LAT, LON = (a.ravel() for a in np.meshgrid(np.arange(box[0] + dlat / 2, box[1], dlat), np.arange(box[2] + dlon / 2, box[3], dlon), indexing="ij"))
            say(f"coarse grid: {len(LAT)} places {spacing:.0f} m apart x {len(shifts)} clock offsets, {len(curves)} curves")
            coarse = run(LAT, LON, shifts, "coarse grid")
            fine_of = lambda q: shifts
        else:
            say(f"{'a large box' if spacing > MAX_GRID_SPACING_M else f'a clock window of +-{clock_s:.0f} s'}: "
                f"places proposed by each curve x each flight's closest approach")
            at = _propose(curves, flights, start, air, ground_h, hm, shifts, wind_mps, box, ex, workers, say, loader, work_dir, info)
            spacing = PROPOSAL_SPACING_M
            coarse = []
            groups = _by_clock([(la, lo, sh0) for la, lo, sh0 in at], CELL_SHIFT_S)
            for i, (items, sh) in enumerate(groups):
                coarse += run(*stack([_grid(la, lo, CELL_M / 2, spacing, box) for la, lo, _ in items]), sh,
                              f"checking proposed places, clock group {i + 1} of {len(groups)} ({len(items)} places)")
            fine_of = lambda q: None
        if not coarse:
            return []
        top = sorted(coarse, key=lambda q: -q["score"])[:N_REFINE]
        say(f"fine grid: {spacing / REFINE_K:.0f} m apart around the {len(top)} best")
        fine = []
        if fine_of(top[0]) is shifts:  # one clock window for all: one pass
            fine = run(*stack([_grid(q["lat"], q["lon"], spacing, spacing / REFINE_K, box) for q in top]), shifts, "fine grid")
        else:  # each best place around its own clock offset, those with close offsets together
            fine_groups = _by_clock([(q["lat"], q["lon"], q["clock_s"]) for q in top], SLACK_S + 2)
            for i, (items, sh) in enumerate(fine_groups):
                fine += run(*stack([_grid(la, lo, spacing, spacing / REFINE_K, box) for la, lo, _ in items]), sh,
                            f"fine grid, clock group {i + 1} of {len(fine_groups)}")
    out, kept = [], []
    fine = sorted(fine, key=lambda q: -q["score"])
    if info is not None and "slice_best" in info and fine:
        info.update(_chance(info.pop("slice_best"), fine[0]))
    for q in fine:
        if len(out) == top_n:
            break
        if any(np.hypot((q["lat"] - a) * M_PER_DEG, (q["lon"] - b) * M_PER_DEG * np.cos(np.radians(a))) < 100.0 for a, b in kept):
            continue  # the same place again (the fine grids overlap)
        kept.append((q["lat"], q["lon"]))
        out.append(q)
    return out
