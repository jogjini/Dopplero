"""Locate a recording from its Doppler curves and ADS-B (review 038; owner's request 2026-11-06).

Pilot scope: public recordings with published locations only.

Two ways to propose where the recorder was, one way to score a place:
- propose_fast: a trusted fit (speed v, slant distance d, closest-approach time t, received) and a
  flight: at emission time t - d/c the aircraft was at P with velocity V; the recorder is at
  distance d from P, on the plane through P perpendicular to V (closest approach, in 3D), at
  ground height: up to two points. Proposals only: they are scored like any other place.
- grid: every point of a search box.
- score_place: every flight's curve as heard from that place (the labeller's emission-time model,
  label.predict_passes) against every detection's traced contours, with ONE clock offset shared
  by all detections (searched), one free source frequency per traced ridge, each flight used by at
  most one detection (assignment). A detection that matches no flight costs nothing. The score is
  the sum over detections of max(0, 1 - median residual / SCALE_CENTS); optionally (experimental)
  minus a cost per flight that should have been heard but wasn't (silence as evidence).
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from . import physics
from . import spectrogram as sg
from .adsb import label as lab
from .adsb.tracks import from_enu, interpolate, segments, stretch_boxes, to_enu

SCALE_CENTS = 16.0        # residual at which a detection-flight pair stops counting
MIN_COVER, MIN_DROP_CENTS = 0.5, 40.0
REACH_S = 300.0           # positions predicted only this far around a flight's close stretches (> HALF_WINDOW_S + travel)
HALF_WINDOW_S = 90.0      # prediction reach around each flight's CPA (as the labeller)
RADIUS_M = 9000.0         # flights predicted within this of the place (audibility, measured ~6-10 km)
CLOCK_S, CLOCK_STEP_S = 15.0, 0.5
# Coordinates, everywhere: WGS84 latitude/longitude; heights above the WGS84 ellipsoid (ADS-B GPS
# altitude is ellipsoidal; ground = elevation above sea level + geoid height); local metres via the
# exact ENU transform (adsb.tracks.to_enu / from_enu); times in UTC unix seconds.


def to_geo(e, n, lat0, lon0, h0=0.0):
    """Inverse of to_local: the point at ellipsoidal height h0 whose local east/north around (lat0,
    lon0, h0) are (e, n). Exact (WGS84); a few iterations because the surface curves away from the
    local plane (~200 m lower at 50 km)."""
    lat, lon, _ = from_enu(float(e), float(n), 0.0, lat0, lon0, h0)
    for _ in range(4):
        ee, nn = to_local(float(lat), float(lon), lat0, lon0, h0)
        dlat, dlon, _ = from_enu(float(e) - ee, float(n) - nn, 0.0, float(lat), float(lon), h0)
        lat, lon = float(dlat), float(dlon)
    return float(lat), float(lon)


def to_local(lat, lon, lat0, lon0, h0=0.0):
    """(lat, lon) at height h0 -> local east/north metres around (lat0, lon0, h0), exact (WGS84)."""
    e = to_enu(np.array([lat]), np.array([lon]), np.array([h0]), lat0, lon0, h0)[0]
    return float(e[0]), float(e[1])


def detection_points(harmonics, g0):
    """(absolute frames from g0 in HOP_S units, cents, ridge index) of a detection's contours."""
    fr = np.concatenate([np.rint(np.array(h["contour"]["t_s"]) / sg.HOP_S).astype(int) for h in harmonics])
    y = np.concatenate([1200 * np.log2(h["contour"]["f_hz"]) for h in harmonics])
    r = np.concatenate([np.full(len(h["contour"]["t_s"]), i) for i, h in enumerate(harmonics)])
    return fr, y, r


def predictions(tracks, lat, lon, h_mic, g0, n_frames, c, ground_h):
    """Predicted cents curves (n_frames, at HOP_S from g0) of every flight within RADIUS_M of the
    place, cut to +-HALF_WINDOW_S around each CPA: (icao, t_cpa_s, cents, {testable, cpa_inside, speed_mps})."""
    frames = sg.frame_times(n_frames)
    ps = lab.predict_passes(tracks, (lat, lon, h_mic), g0, n_frames * sg.HOP_S, frames, c, ground_h, RADIUS_M, reach_s=REACH_S)
    out = []
    for p in ps:
        if p["on_ground_at_cpa"]:
            continue
        cents = 1200 * np.log2(p["ratio"])
        cents[np.abs(frames - p["t_cpa_s"]) > HALF_WINDOW_S] = np.nan
        # testable: the labeller's rule (>= 30 s, >= 200 cents); cpa_inside: ADS-B samples around the CPA
        out.append((p["icao"], p["t_cpa_s"], cents, {"testable": lab.testable(p["ratio"]), "cpa_inside": p["cpa_inside_track"],
                                                     "speed_mps": p["speed_cpa_mps"]}))
    return out


def residual(fr, y, r, pred, shift_frames):
    """Median |cents| of the detection against pred delayed by shift_frames, one free offset per
    ridge; inf if the prediction covers too little of it or is too flat there."""
    j = fr - shift_frames
    ok = (j >= 0) & (j < len(pred))
    p = np.full(len(fr), np.nan)
    p[ok] = pred[j[ok]]
    ok &= np.isfinite(p)
    if ok.mean() < MIN_COVER or np.ptp(p[ok]) < MIN_DROP_CENTS:
        return np.inf
    e, rr = y[ok] - p[ok], r[ok]
    for k in np.unique(rr):
        e[rr == k] -= np.median(e[rr == k])
    return float(np.median(np.abs(e)))


def eligible(preds, k, valid=None, edge_s=60.0):
    """Which predicted passes we could have heard at clock shift k (frames): testable by the
    labeller's rule (review 042), CPA supported by ADS-B samples, closest approach with +-edge_s of
    valid audio around it after the shift, one pass per aircraft (segments of one flight with CPAs
    < 120 s apart count once)."""
    out, seen = [], {}
    for icao, t_cpa, cents, info in preds:
        n = len(cents)
        c, w = int(round(t_cpa / sg.HOP_S)) + k, int(round(edge_s / sg.HOP_S))
        ok = info["testable"] and info["cpa_inside"] and 0 <= c - w and c + w < n and (valid is None or bool(np.all(valid[c - w:c + w + 1])))
        if ok and any(abs(t_cpa - t) < 120 for t in seen.get(icao, [])):
            ok = False
        if ok:
            seen.setdefault(icao, []).append(t_cpa)
        out.append(ok)
    return np.array(out, bool)


def car_like(fit, max_speed=25.0, max_dist=150.0):
    """Our road-traffic rule for a fit: median speed below max_speed AND distance below max_dist."""
    return bool(fit and fit["speed_mps"]["median"] < max_speed and fit["cpa_distance_m"]["median"] < max_dist)


MIN_FLYING_MPS = 20.0  # a heuristic (review 044): ground speed at CPA; slow aircraft, helicopters or a strong headwind can be below it


def score_place(dets, preds, miss=0.0, valid=None, flying_only=False):
    """(score, clock offset s, [(detection index, flight icao, residual)], [unheard icaos]) at one
    place, with one clock offset shared by every detection (searched over +-CLOCK_S).
    miss > 0 (experimental, review 041): each eligible() pass left unheard costs miss; the
    assignment maximises matches + miss * (eligible passes heard), so the penalty shapes the pairing.
    A place that explains no detection can't win: it scores 0 with miss = 0 and -miss * eligible
    passes otherwise, and callers rank places with matches first.
    flying_only (review 044, a heuristic): only targets whose ground speed at closest approach is
    >= MIN_FLYING_MPS can explain a detection: a near-stationary transponder predicts a near-flat
    curve that fits any steady tone (the owner's fake-sky winner: targets at 0-10 m/s). It can also
    exclude a genuinely slow source; the job records what it excluded at its top places. The labeller's testability judged on the
    compared overlap was tried first and dropped: real traced stretches are often < 30 s or < 200 c."""
    if flying_only:
        preds = [q for q in preds if q[3]["speed_mps"] >= MIN_FLYING_MPS]
    if not dets or not preds:
        return 0.0, 0.0, [], []
    shifts = np.arange(-CLOCK_S, CLOCK_S + 1e-9, CLOCK_STEP_S)
    # a detection is one set of contours, or a duplicate group (a list of sets): its best member counts.
    # Only member-pass pairs whose frames can overlap at some shift are compared: any other pair
    # covers less than MIN_COVER at every shift, so residual() is inf for it (hours-long recordings:
    # most pairs are hours apart)
    K = int(round(CLOCK_S / sg.HOP_S)) + 1
    span = [(f[0], f[-1]) if len(f) else (1, 0) for f in (np.flatnonzero(np.isfinite(p[2])) for p in preds)]
    pairs = []
    for i, d in enumerate(dets):
        qs = [q for q in (d if isinstance(d, list) else [d]) if len(q[0])]
        for j, (a, b) in enumerate(span):
            near = [q for q in qs if q[0].min() - K <= b and q[0].max() + K >= a]
            if near:
                pairs.append((i, j, near))
    best = None
    for s in shifts:
        k = int(round(s / sg.HOP_S))
        m = np.full((len(dets), len(preds)), np.inf)
        for i, j, near in pairs:
            m[i, j] = min(residual(*q, preds[j][2], k) for q in near)
        val = np.clip(1 - m / SCALE_CENTS, 0, None)
        el = eligible(preds, k, valid) if miss > 0 else np.zeros(len(preds), bool)
        ri, ci = linear_sum_assignment(-(val + miss * el[None, :] * (val > 0)))
        heard = [(i, j) for i, j in zip(ri, ci) if val[i, j] > 0]
        got = {preds[j][0] for _, j in heard}  # any heard segment of a flight means it was heard
        unheard = sorted({preds[j][0] for j in np.flatnonzero(el)} - got)
        total = float(sum(val[i, j] for i, j in heard)) - miss * len(unheard)
        key = (bool(heard), total)
        if best is None or key > best[0]:
            best = (key, (total, float(s), [(int(i), preds[j][0], float(m[i, j])) for i, j in heard], unheard))
    return best[1]


def _state(tr, idx, times, frame, h_ground):
    """Positions and velocities (len(times), 3) of one segment in frame = (lat, lon, h), NaN outside:
    interpolate() smooths over 1 s, which is only right on a sorted, regular grid with seconds of margin
    (3 points 0.5 s apart give speeds ~14 % low), so it runs on such a grid and is read off at times."""
    g = np.arange(times.min() - 5.0, times.max() + 5.0, 0.25)
    q = interpolate(tr, idx, g, frame, h_ground)
    at = lambda tt: np.stack([np.interp(tt, g, q[:, k], left=np.nan, right=np.nan) for k in range(3)], 1)
    return at(times), at(times + 0.5) - at(times - 0.5)


def _offsets(P, V, d):
    """Horizontal offsets (N, 2 sides, 2) from aircraft at P (N, 3; a local frame whose z = 0 is the
    recorder's height) to the up-to-two recorder places at slant distance d (N,), in the plane
    perpendicular to the velocity V (N, 3); NaN where there is none."""
    uz = -P[:, 2]  # u = O - P with u_z fixed: Vx ux + Vy uy = -Vz uz ; ux^2 + uy^2 = d^2 - uz^2
    h = np.hypot(V[:, 0], V[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        a = V[:, :2] / h[:, None]
        q = -V[:, 2] * uz / h  # a . (ux, uy) = q
        rho2 = d * d - uz * uz
        rest = rho2 - q * q
        root = np.where((rho2 > 0) & (h > 1e-6) & (rest >= 0), np.sqrt(np.abs(rest)), np.nan)
    perp = np.stack([-a[:, 1], a[:, 0]], 1)
    return np.stack([q[:, None] * a + sgn * root[:, None] * perp for sgn in (1, -1)], 1)


def propose_fast(fits, tracks, lat0, lon0, h_ground, h_mic, g0, c):
    """Candidate places from trusted fits: for each fit (t_cpa received, absolute; speed; slant
    distance) and each flight whose 3D speed at emission time is within the fit's speed range, the
    up-to-two points at the recorder's height (h_mic, ellipsoidal) at slant distance d from the
    aircraft, in the plane perpendicular to its velocity. The geometry is solved in a local frame
    under the aircraft (earth curvature over d <= ~20 km: < 35 m), and every candidate is returned
    as exact WGS84 lat/lon plus local e, n metres around (lat0, lon0)."""
    out = []
    for f in fits:
        t_r, v_lo, v_hi = f["t_cpa_unix"], f["speed_p05"], f["speed_p95"]
        d = f["distance"]
        t_e = t_r - d / c
        for tr in tracks:
            for idx in segments(tr):
                if not tr.t[idx[0]] + 1 <= t_e <= tr.t[idx[-1]] - 1:
                    continue
                # the aircraft's position, then a frame at the recorder's height right under it
                pos0 = _state(tr, idx, np.array([t_e]), (lat0, lon0, h_mic), h_ground)[0][0]
                if not np.isfinite(pos0).all():
                    continue
                alat, alon, _ = from_enu(pos0[0], pos0[1], 0.0, lat0, lon0, h_mic)
                alat, alon = float(alat), float(alon)
                P, V = (x[0] for x in _state(tr, idx, np.array([t_e]), (alat, alon, h_mic), h_ground))
                if not (np.isfinite(P).all() and np.isfinite(V).all()):
                    continue
                if not v_lo <= np.linalg.norm(V) <= v_hi:
                    continue
                for u in _offsets(P[None], V[None], np.array([d]))[0]:
                    if not np.isfinite(u).all():
                        continue
                    la, lo = to_geo(P[0] + u[0], P[1] + u[1], alat, alon, h_mic)
                    e, nn = to_local(la, lo, lat0, lon0, h_mic)
                    out.append({"lat": la, "lon": lo, "e": e, "n": nn, "icao": tr.icao, "fit": f["id"]})
    return out


def track_boxes(tr, lat0, lon0, h0, t0, t1, margin_m=100.0, smooth_s=1.0):
    """(m, 4) boxes [e0, e1, n0, n1] (local metres around (lat0, lon0) at height h0) that provably
    contain everything interpolate() can place this track at during [t0, t1]: one per stretch between
    consecutive samples of a segment (interpolation never crosses a segment cut), widened by
      - the cubic Hermite bound: |H - chord| <= 0.0963 |dp| + 4/27 h (|v0| + |v1|), and
      - the smoothing reach: a Gaussian of smooth_s truncated at 4 sigma (reflection-padded at the
        ends) moves a point by at most (4 smooth_s + 0.1 s) * max |H'|, max |H'| <= 1.5 |dp| / h + |v0| + |v1|,
    with |v| bounded by the larger of the reported and finite-difference velocity (interpolate uses
    one or the other) and margin_m for frame and sub-point approximations (metres over <= 20 km)."""
    out = [stretch_boxes(tr, idx, (lat0, lon0, h0), t0, t1, margin_m, smooth_s)[0] for idx in segments(tr)]
    out = [b for b in out if len(b)]
    return np.concatenate(out) if out else np.zeros((0, 4))


class TileIndex:
    """Which tracks can matter at a place: each track registered in every square tile (local metres
    around an origin) that one of its track_boxes touches. A place within RADIUS_M of any point the
    track can be interpolated at lies in that point's tile or a neighbour (tile_m >= RADIUS_M), so its
    tile and the 8 neighbours give a superset of the tracks predictions() can return there. Build it
    over the widest time window used (fake skies shift times only)."""

    def __init__(self, tracks, lat0, lon0, h0, t0, t1, tile_m=20000.0):
        assert RADIUS_M <= tile_m, "a tile and its neighbours must cover the prediction radius"
        self.lat0, self.lon0, self.h0, self.tile = lat0, lon0, h0, tile_m
        self.cells = {}
        for k, tr in enumerate(tracks):
            bx = np.floor(track_boxes(tr, lat0, lon0, h0, t0, t1) / tile_m).astype(int)
            for a0, a1, b0, b1 in bx:
                for a in range(a0, a1 + 1):
                    for b in range(b0, b1 + 1):
                        self.cells.setdefault((a, b), set()).add(k)

    def key(self, lat, lon):
        """The tile holding (lat, lon): places with the same key get the same near() tracks."""
        e, n = to_local(lat, lon, self.lat0, self.lon0, self.h0)
        return int(np.floor(e / self.tile)), int(np.floor(n / self.tile))

    def near(self, lat, lon):
        """Indices of the tracks that can be within the prediction radius of (lat, lon)."""
        a, b = self.key(lat, lon)
        return sorted(set().union(*(self.cells.get((a + i, b + j), set()) for i in (-1, 0, 1) for j in (-1, 0, 1))))


def propose_joint(fits, tracks, lat0, lon0, h_ground, h_mic, c, clock_s, step_s=0.5, n_d=17, tol_m=500.0, min_sep_s=30.0,
                  spacing_m=250.0):
    """Candidate places explained by two trusted fits at once, with ONE clock offset for both (the
    recording's clock may be off by up to clock_s; propose_fast assumes none). For every offset on a
    step_s grid and n_d distances across each fit's range (distance_p05..p95), each fit and each flight
    of matching speed give the recorder places on two lines perpendicular to the flight; where lines
    of two fits on two different aircraft meet within tol_m at the same offset, the meeting points
    are proposed (one per offset, thinned to spacing_m). Two fits count as different planes only from different duplicate groups with closest
    approaches more than min_sep_s apart. Proposals only: scored like any other candidate."""
    deltas = np.arange(-clock_s, clock_s + 1e-9, step_s)
    lines = []  # (fit position k, icao, points (len(deltas), n_d, 3) in the job's ENU frame)
    for k, f in enumerate(fits):
        ds = np.linspace(f["distance_p05"], f["distance_p95"], n_d)
        te = (f["t_cpa_unix"] - deltas[:, None] - ds[None, :] / c).ravel()  # clock reads late by delta
        dd = np.broadcast_to(ds, (len(deltas), n_d)).ravel()
        for tr in tracks:
            for idx in segments(tr):
                if te.max() < tr.t[idx[0]] + 1 or te.min() > tr.t[idx[-1]] - 1:
                    continue
                mid = _state(tr, idx, np.array([np.clip(np.median(te), tr.t[idx[0]], tr.t[idx[-1]])]), (lat0, lon0, h_mic), h_ground)[0][0]
                if not np.isfinite(mid).all():
                    continue
                alat, alon, _ = from_enu(mid[0], mid[1], 0.0, lat0, lon0, h_mic)
                alat, alon = float(alat), float(alon)  # a frame at the recorder's height under the aircraft
                P, V = _state(tr, idx, te, (alat, alon, h_mic), h_ground)
                with np.errstate(invalid="ignore"):
                    sp = np.linalg.norm(V, axis=1)
                if not np.any((sp >= f["speed_p05"]) & (sp <= f["speed_p95"])):
                    continue
                u = _offsets(P, V, dd)
                u[~((sp >= f["speed_p05"]) & (sp <= f["speed_p95"]))] = np.nan
                for side in (0, 1):
                    xy = P[:, :2] + u[:, side]
                    if not np.isfinite(xy).any():
                        continue
                    la, lo, hh = from_enu(xy[:, 0], xy[:, 1], np.zeros(len(xy)), alat, alon, h_mic)
                    lines.append((k, tr.icao, to_enu(la, lo, hh, lat0, lon0, h_mic).reshape(len(deltas), n_d, 3)))
    # shortcut: every pair of lines is bbox-tested (quadratic in lines); bin them by tile if a huge box makes this slow
    lo_ = np.array([np.nanmin(x.reshape(-1, 3), 0) for _, _, x in lines]) if lines else np.zeros((0, 3))
    hi_ = np.array([np.nanmax(x.reshape(-1, 3), 0) for _, _, x in lines]) if lines else np.zeros((0, 3))
    out = []
    for i in range(len(lines)):
        ki, ici, xi = lines[i]
        for j in range(i + 1, len(lines)):
            kj, icj, xj = lines[j]
            fi, fj = fits[ki], fits[kj]
            if ki == kj or ici == icj or fi["group"] == fj["group"] or abs(fi["t_cpa_unix"] - fj["t_cpa_unix"]) <= min_sep_s:
                continue
            if np.any(lo_[i, :2] > hi_[j, :2] + tol_m) or np.any(lo_[j, :2] > hi_[i, :2] + tol_m):
                continue
            gap = np.linalg.norm(xi[:, :, None] - xj[:, None, :], axis=-1)  # (offset, d_i, d_j)
            if not np.isfinite(gap).any() or np.nanmin(gap) > tol_m:
                continue
            # two lines cross at nearly every offset: the fits' distance ranges only bound the offset,
            # so the meeting points along the offsets are all proposed (>= spacing_m apart), not one
            last = None
            for a in range(len(deltas)):
                if not np.isfinite(gap[a]).any() or np.nanmin(gap[a]) > tol_m:
                    continue
                b, cc = np.unravel_index(np.nanargmin(gap[a]), gap[a].shape)
                m = (xi[a, b] + xj[a, cc]) / 2
                if last is not None and np.linalg.norm(m - last) < spacing_m:
                    continue
                last = m
                la, lo = to_geo(m[0], m[1], lat0, lon0, h_mic)
                e, nn = to_local(la, lo, lat0, lon0, h_mic)
                out.append({"lat": la, "lon": lo, "e": e, "n": nn, "icao": [ici, icj], "fit": [fi["id"], fj["id"]],
                            "clock_s": float(deltas[a]), "gap_m": float(gap[a, b, cc])})
    return out
