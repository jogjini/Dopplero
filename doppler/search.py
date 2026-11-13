"""Place search from traced curves x real flight tracks (lead's plan step 3, reviews 052; first in
scripts/place_search.py, now shared with the app's job). Development-stage method.

1. Proposals: each ridge (a detection's traced tones) x each propeller flight (piston, turboprop, unknown type)
   airborne at the ridge's time: the flight's straight-pass curve with its ADS-B ground speed fixed is fitted for
   the heard closest-approach time and the slant distance (the best 3 distances, each as a band); every (time,
   distance) and every clock offset in the prior puts the listener at one of two places across the flight's real
   path (`places_line`). Places are pooled in 1 km cells x 10 s clock buckets. How well one flight fits one ridge
   does not tell flights apart (on the video ~130 propeller flights fit each ridge about as well), so every flight
   is kept; what singles out a place is several flights agreeing at one clock.
2. A cheap first pass (`stage1`) orders the cells backed by two or more flights by their proposing flights' evidence
   near the cell's own clock; the best N_VERIFY cells (then the best single-flight ones) get
3. the full check (`verify`): every flight within 9 km predicted from its real track and scored on the recording's
   tone map at every clock offset of the prior, each against N_NULL seeded far offsets (its chance level); place
   score T_cap = sum over flights of min(CAP, max(0, E - ln n)), its best over the clock window.
Validation so far (development, PROGRESS 2026-11-10): the YouTube camera at 0.8 km with a known clock (+-2 s),
fake sky clearly lower; with an unknown clock (+-180 s) at fake-sky level.
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import evidence as ev
from . import locate, physics
from .adsb.tracks import segments
from .adsb.types import aircraft_class

D_GRID = 200.0 * 1.06 ** np.arange(66)  # 0.2 .. ~9 km, 6 % apart
T_HALF_S, T_STEP_S = 60.0, 0.5           # heard-CPA times tried around the ridge's steepest fall
N_KEEP = 3                               # distances kept per (ridge, flight): the best local minima
BAND = (0.85, 0.92, 1.0, 1.08, 1.15)     # each fitted distance proposed as a band: distance errors grow with distance
CLOCK_STEP_S = 5.0                       # step of the proposal lines over the clock prior
CELL_M, BUCKET_S = 1000.0, 10.0          # pooling of proposed places
N_VERIFY, N_NULL, SEED = 1500, 2000, 52  # 2000 far offsets: rankings as with every far offset (rank corr. 0.98-0.99); 400 shuffled the top
GUARD_S = 600.0                          # far offsets (chance level) start beyond this
STAGE1_NULL, STAGE1_HALF_S = 400, 10.0   # first pass: a rough chance level, the cell's own clock bucket +- 10 s
# P(a pass of this class at this closest distance is heard at all): the labeller's confirmed rates at 10 known sites
# (LEAD_REVIEW §1.6), by distance bins 0-1.5, 1.5-3, 3-6, 6-9, 9+ km; unknown types as pistons
HEARD = {"piston": (0.58, 0.68, 0.51, 0.15, 0.02), "turboprop": (0.6, 0.6, 0.6, 0.6, 0.13),
         "jet": (0.22, 0.06, 0.08, 0.02, 0.01), "helicopter": (0.05,) * 5}
HEARD_BINS = (1500.0, 3000.0, 6000.0, 9000.0)
CAP = 8.0  # most one pass may add (nats): decoy peaks are single flights (LEAD_REVIEW §4.1)
G = {}


def is_prop(type_):
    return aircraft_class(type_) in ("piston", "turboprop", "unknown")


def init(map_dir, tracks, frame, hg, t0, c, clock_s, n_null=N_NULL, full=False, null_all=False):
    """Worker state for stage1 / verify: the recording's tone map (map_dir/map.npy, valid.npy) wrapped for clock
    offsets, the flights and their tile index. full / null_all: the full profile instead of the window + far
    offsets, and every far offset as the chance level (slow; kept to check the fast path)."""
    from pathlib import Path
    valid = np.load(Path(map_dir) / "valid.npy")
    dur = len(valid) * ev.STEP_S
    S = int(min(1800.0, dur / 2) / ev.STEP_S)
    Mp, vp = ev.wrap(np.load(Path(map_dir) / "map.npy"), valid, S)
    G.update(Mp=Mp, vp=vp, S=S, dur=dur, tracks=tracks, by_icao={tr.icao: tr for tr in tracks}, frame=frame, hg=hg, t0=t0, c=c,
             clock_s=clock_s, n_null=n_null, fast=not full, null_all=null_all,
             idx=locate.TileIndex(tracks, frame[0], frame[1], hg + 1.5, t0 - 1800, t0 + dur + 1800))


def ridge_points(contours):
    """(t, cents, tone index) of a ridge's traced tones; contours [{"t", "f"}] in absolute seconds and Hz."""
    t = np.concatenate([np.asarray(q["t"], float) for q in contours])
    y = np.concatenate([1200 * np.log2(np.asarray(q["f"], float)) for q in contours])
    g = np.concatenate([np.full(len(q["t"]), i) for i, q in enumerate(contours)])
    return t, y, g


def steepest(t, y, g):
    """Time of the steepest fall of the longest tone (a first guess of the heard closest approach)."""
    k = np.bincount(g).argmax()
    tt, yy = t[g == k], y[g == k]
    o = np.argsort(tt)
    tt, yy = tt[o], yy[o]
    if len(tt) < 5:
        return float(np.median(t))
    sl = np.gradient(yy, tt)
    return float(tt[np.argmin(np.convolve(sl, np.ones(5) / 5, mode="same"))])


def fit_fixed_v(t, y, g, v, c, t_mid):
    """[(residual cents, heard CPA time, distance)] for the N_KEEP best distances: per distance the best time on
    the grid, one free pitch offset per tone (its median), residual = median |cents| over all points."""
    ts = t_mid + np.arange(-T_HALF_S, T_HALF_S + 1e-9, T_STEP_S)
    best = []
    for d in D_GRID:
        m = 1200 * np.log2(physics.straight_pass(t[None, :], ts[:, None], v, d, c)[0])  # (times, points)
        r = y[None, :] - m
        for k in np.unique(g):
            r[:, g == k] -= np.median(r[:, g == k], axis=1, keepdims=True)
        res = np.median(np.abs(r), axis=1)
        i = int(np.argmin(res))
        best.append((float(res[i]), float(ts[i]), float(d)))
    res = np.array([b[0] for b in best])
    loc = [i for i in range(len(res)) if (i == 0 or res[i] <= res[i - 1]) and (i == len(res) - 1 or res[i] <= res[i + 1])]
    return [best[i] for i in sorted(loc, key=lambda i: res[i])[:N_KEEP]]


def places(tr, t_cpa_heard, d, clock_s, frame, h_ground, c):
    """The up-to-two listener places (e, n in frame, metres) for one (heard CPA time, distance, clock offset):
    the aircraft's real position and velocity when it emitted the closest-approach sound."""
    t_e = t_cpa_heard - clock_s - d / c
    for idx in segments(tr):
        if tr.t[idx[0]] + 1 <= t_e <= tr.t[idx[-1]] - 1:
            P, V = (a[0] for a in locate._state(tr, idx, np.array([t_e]), frame, h_ground))
            if not (np.isfinite(P).all() and np.isfinite(V).all()):
                return []
            return [P[:2] + u for u in locate._offsets(P[None], V[None], np.array([d]))[0] if np.isfinite(u).all()]
    return []


def places_line(tr, t_cpa_heard, d, clocks, frame, h_ground, c):
    """places() for many clock offsets at once: (clock offsets kept, places (n, 2 sides, 2) e/n metres, NaN where
    a side has no solution) from one _state call per track segment."""
    t_e = t_cpa_heard - clocks - d / c
    out_k, out_p = [], []
    for idx in segments(tr):
        m = (t_e >= tr.t[idx[0]] + 1) & (t_e <= tr.t[idx[-1]] - 1)
        if not m.any():
            continue
        P, V = locate._state(tr, idx, t_e[m], frame, h_ground)
        ok = np.isfinite(P).all(axis=1) & np.isfinite(V).all(axis=1)
        if not ok.any():
            continue
        U = locate._offsets(P[ok], V[ok], np.full(ok.sum(), d))  # (n, 2, 2)
        out_k.append(clocks[m][ok])
        out_p.append(P[ok][:, None, :2] + U)
    if not out_k:
        return np.zeros(0), np.zeros((0, 2, 2))
    return np.concatenate(out_k), np.concatenate(out_p)


def speed_at(tr, t, frame, h_ground):
    for idx in segments(tr):
        if tr.t[idx[0]] + 1 <= t <= tr.t[idx[-1]] - 1:
            V = locate._state(tr, idx, np.array([t]), frame, h_ground)[1][0]
            return float(np.linalg.norm(V)) if np.isfinite(V).all() else None
    return None


def turn_deg(tr, t, half_s=60.0):
    """Heading change of the track over +-half_s around t (degrees), from the ADS-B reported track angle."""
    k = (tr.t >= t - half_s) & (tr.t <= t + half_s) & np.isfinite(tr.track)
    if k.sum() < 2:
        return None
    a = np.unwrap(np.radians(tr.track[k]))
    return float(np.degrees(a.max() - a.min()))


def propose(ridges, tracks, t0, frame, hg, c, box_local, clock_s, clock_step_s, say=None):
    """Proposal cells {(ix, iy, clock bucket): {"flights": set, "ridges": set, "res": best residual}} from every
    ridge x propeller flight; ridges: [(id, t abs, cents, tone)]."""
    cells = {}
    for n_done, (rid, t, y, g) in enumerate(ridges):
        if say:
            say("proposing", n_done, len(ridges))
        t_mid = steepest(t, y, g)
        cand = []
        for tr in tracks:
            v = speed_at(tr, t_mid - 10.0, frame, hg)
            if not v or v < locate.MIN_FLYING_MPS:
                continue
            fits = fit_fixed_v(t, y, g, v, c, t_mid)
            if fits:
                cand.append((fits[0][0], tr, fits))
        clocks = np.arange(-clock_s, clock_s + 1e-9, clock_step_s)
        for res0, tr, fits in cand:
            for res, tc, d in fits:
                for b in BAND:
                    ks, P = places_line(tr, tc, d * b, clocks, frame, hg, c)
                    for side in range(P.shape[1] if len(P) else 0):
                        xy = P[:, side]
                        m = np.isfinite(xy).all(axis=1) & (xy[:, 0] >= box_local[0]) & (xy[:, 0] <= box_local[1]) & \
                            (xy[:, 1] >= box_local[2]) & (xy[:, 1] <= box_local[3])
                        for (ex, ny), k in zip(xy[m], ks[m]):
                            key = (int(ex // CELL_M), int(ny // CELL_M), int(round(k / BUCKET_S)))
                            q = cells.setdefault(key, {"flights": set(), "ridges": set(), "res": np.inf})
                            q["flights"].add(tr.icao)
                            q["ridges"].add(rid)
                            q["res"] = min(q["res"], res)
    return cells


def verify(pt):
    """(best T over the clock window, its clock offset s, n passes, heard [(icao, type, km, evidence)]) at one place:
    each pass's z at every window offset and at N_NULL seeded far offsets (same for every place and sky)."""
    S = G["S"]
    lat, lon = pt
    trs = [G["tracks"][i] for i in G["idx"].near(lat, lon)]
    ps = ev.passes(trs, (lat, lon), G["hg"], G["t0"], G["dur"], G["c"])
    if not ps:
        return 0.0, None, 0, [], None
    rng = np.random.default_rng(SEED)
    nn = G.get("n_null", N_NULL)
    far = rng.integers(int(600 / ev.STEP_S), S, nn) * rng.choice((-1, 1), nn)
    win = np.arange(-int(G["clock_s"] / ev.STEP_S), int(G["clock_s"] / ev.STEP_S) + 1)
    E = np.zeros((len(ps), len(win)), np.float32)
    for i, p in enumerate(ps):
        if G.get("fast"):  # only the window and the N_NULL far offsets (review 052: compared with the full profile first)
            zz = ev.profile_at(G["Mp"], G["vp"], p, S, np.concatenate([win, far]))
            zw_raw, zf = zz[:len(win)], zz[len(win):]
        else:  # full profile, shared with site_evidence; with null_all its chance level uses every far offset
            z = ev.profile(G["Mp"], G["vp"], p, S)
            far_all = np.flatnonzero(np.abs(np.arange(-S, S + 1)) >= int(600 / ev.STEP_S)) - S
            zw_raw, zf = z[S + win], z[S + (far_all if G.get("null_all") else far)]
        null = np.sort(zf[np.isfinite(zf)])
        zw = np.nan_to_num(zw_raw, nan=-np.inf)
        zw = np.maximum.reduce([np.roll(zw, k) for k in range(-8, 9)]) if len(zw) > 17 else np.full(len(zw), zw.max())  # +-4 s, as site_evidence
        if len(null) < 100:
            continue
        q = null[int(0.95 * len(null))]
        over = null[null > q] - q
        scale = over.mean() if len(over) else 1.0
        with np.errstate(over="ignore"):
            pv = np.where(zw > q, 0.05 * np.exp(-(zw - q) / scale), np.maximum(1 - np.searchsorted(null, zw) / len(null), 0.05))
        E[i] = -np.log(np.maximum(pv, 1e-300))
    T = np.clip(E - np.log(len(ps)), 0, None).sum(axis=0)
    k = int(np.argmax(T))
    heard = sorted(((p["icao"], p["type"], round(p["d_cpa_m"] / 1000, 1), round(float(E[i, k] - np.log(len(ps))), 1))
                    for i, p in enumerate(ps) if E[i, k] > np.log(len(ps))), key=lambda h: -h[3])
    # every pass's evidence over the whole clock window, so other place statistics can be computed without a re-run
    keep = {"E": E.astype(np.float16), "type": [p["type"] for p in ps], "d_m": [float(p["d_cpa_m"]) for p in ps],
            "icao": [p["icao"] for p in ps]}
    return float(T[k]), float(win[k] * ev.STEP_S), len(ps), heard, keep


def stage1(item):
    """Cheap first-pass score of one proposal cell: only the flights that proposed it, at its centre, near its own
    clock; sum over them of min(CAP, max(0, E - ln k)), k = those flights (rough: for ordering cells only)."""
    S = G["S"]
    lat, lon, icaos, clock = item
    trs = [G["by_icao"][i] for i in icaos if i in G["by_icao"]]
    ps = ev.passes(trs, (lat, lon), G["hg"], G["t0"], G["dur"], G["c"]) if trs else []
    if not ps:
        return 0.0
    rng = np.random.default_rng(SEED)
    far = rng.integers(int(600 / ev.STEP_S), S, STAGE1_NULL) * rng.choice((-1, 1), STAGE1_NULL)
    c0 = int(round(clock / ev.STEP_S))
    win = np.arange(c0 - int(STAGE1_HALF_S / ev.STEP_S), c0 + int(STAGE1_HALF_S / ev.STEP_S) + 1)
    E = []
    for p in ps:
        zz = ev.profile_at(G["Mp"], G["vp"], p, S, np.concatenate([win, far]))
        zw, zf = np.nan_to_num(zz[:len(win)], nan=-np.inf), np.sort(zz[len(win):][np.isfinite(zz[len(win):])])
        if len(zf) < 100:
            E.append(np.zeros(len(win)))
            continue
        pv = np.maximum(1 - np.searchsorted(zf, zw) / len(zf), 1.0 / len(zf))
        E.append(-np.log(pv))
    X = np.clip(np.array(E) - np.log(len(ps)), 0, CAP)
    return float(X.sum(axis=0).max())


def statistics(rec, table=None):
    """Place statistics over the clock window from one place's saved evidence: current (sum of E - ln n),
    capped (each pass at most CAP), class-weighted (sum of ln(1 - pi + pi e^X), X = max(0, E - ln n),
    pi = P(heard | class, distance)),
    and both. Each as its best value over the window."""
    if rec is None:
        return {"T": 0.0, "T_cap": 0.0, "T_class": 0.0, "T_class_cap": 0.0}
    E = rec["E"].astype(np.float32)
    n = len(E)
    pi = np.array([HEARD.get(aircraft_class(t, table), HEARD["piston"])[int(np.searchsorted(HEARD_BINS, d))] for t, d in zip(rec["type"], rec["d_m"])])
    X = np.clip(E - np.log(n), 0, None)
    W = np.log1p(pi[:, None] * np.expm1(np.minimum(X, 60.0)))  # beyond what n passes' chances explain, as T
    return {"T": float(X.sum(axis=0).max()), "T_cap": float(np.minimum(X, CAP).sum(axis=0).max()),
            "T_class": float(W.sum(axis=0).max()), "T_class_cap": float(np.minimum(W, CAP).sum(axis=0).max())}


def prop_statistics(rec, table, limits, silence):
    """The place statistic on propeller passes only (piston, turboprop) within a class's distance limit
    (limits: {class: max closest distance m}), n = passes kept; with silence, each kept pass not heard at a clock
    offset (no evidence beyond chance there) costs ln(1 - pi), pi = P(heard | class, distance) (bounded: a silent
    close piston costs ~1.1, nothing else is counted)."""
    if rec is None:
        return 0.0
    cls = [aircraft_class(t, table) for t in rec["type"]]
    keep = [i for i, (c, d) in enumerate(zip(cls, rec["d_m"])) if c in limits and d <= limits[c]]
    if not keep:
        return 0.0
    E = rec["E"].astype(np.float32)[keep]
    X = np.clip(E - np.log(len(keep)), 0, None)
    T = X.sum(axis=0)
    if silence:
        pi = np.array([HEARD[cls[i]][int(np.searchsorted(HEARD_BINS, rec["d_m"][i]))] for i in keep])
        T = T + ((X <= 0) * np.log1p(-pi)[:, None]).sum(axis=0)
    return float(T.max())



def search_places(map_dir, ridges, tracks, t0, frame, hg, c, box, clock_s, workers=4, n_verify=N_VERIFY, say=None):
    """The whole search for one sky: proposals from ridges [(id, t abs s, cents, tone)] x the propeller flights of
    tracks, the first pass over two-flight cells, the full check of the best n_verify places. Returns
    [{lat, lon, score (T_cap), T, clock_s, n_passes, heard [(icao, type, km, nats)], two_flight}] best first,
    and the per-place evidence records (for other statistics without a re-run)."""
    lat0, lon0 = frame[0], frame[1]
    e0, n0 = locate.to_local(box[0], box[2], lat0, lon0)
    e1, n1 = locate.to_local(box[1], box[3], lat0, lon0)
    step = min(CLOCK_STEP_S, max(clock_s, 0.5))
    props = [tr for tr in tracks if is_prop(tr.type)]
    cells = propose(ridges, props, t0, frame, hg, c, (e0, e1, n0, n1), clock_s, step, say)
    multi = sorted((k for k, q in cells.items() if len(q["flights"]) >= 2), key=lambda k: (-len(cells[k]["flights"]), cells[k]["res"]))
    single = sorted((k for k, q in cells.items() if len(q["flights"]) == 1), key=lambda k: cells[k]["res"])
    with ProcessPoolExecutor(workers, initializer=init, initargs=(map_dir, tracks, frame, hg, t0, c, clock_s)) as ex:
        items = [(*locate.to_geo((k[0] + 0.5) * CELL_M, (k[1] + 0.5) * CELL_M, lat0, lon0), sorted(cells[k]["flights"]), k[2] * BUCKET_S) for k in multi]
        first = []
        for r in ex.map(stage1, items, chunksize=32):
            first.append(r)
            if say and len(first) % 500 == 0:
                say("first pass", len(first), len(items))
        multi = [multi[i] for i in np.argsort(-np.array(first), kind="stable")]
        order, seen = [], set()
        for k in multi + single:  # one full check per 1 km cell (the window covers every clock)
            if (k[0], k[1]) not in seen:
                seen.add((k[0], k[1]))
                order.append(k)
            if len(order) == n_verify:
                break
        pts = [locate.to_geo((k[0] + 0.5) * CELL_M, (k[1] + 0.5) * CELL_M, lat0, lon0) for k in order]
        res = []
        for r in ex.map(verify, pts, chunksize=8):
            res.append(r)
            if say and len(res) % 50 == 0:
                say("checking places", len(res), len(pts))
    out = []
    for p, k, r in zip(pts, order, res):
        st = statistics(r[4])
        out.append({"lat": p[0], "lon": p[1], "score": st["T_cap"], "T": st["T"], "clock_s": r[1], "n_passes": r[2], "heard": r[3][:8],
                    "two_flight": len(cells[k]["flights"]) >= 2})
    o = sorted(range(len(out)), key=lambda i: -out[i]["score"])
    return [out[i] for i in o], [res[i][4] for i in o], {"n_cells": len(cells), "n_two_flight": len(multi), "n_verified": len(pts)}
