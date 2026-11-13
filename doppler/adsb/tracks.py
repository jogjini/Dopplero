"""Load ADS-B data into one track format, and turn tracks into positions around the mic.

Source: adsb.lol's public daily archive (github.com/adsblol/globe_history_YYYY, ODbL), which
stores one readsb "trace" per aircraft. The same trace format is written by a readsb
receiver of your own, so `parse_readsb_trace` covers both.

A day is ~4 GB, so we stream it and keep only aircraft that come within `radius_km` of the
site at any time that day; the result (a few MB) is cached in data/adsb/.
"""
import gzip
import io
import json
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.ndimage import gaussian_filter1d

FT, KT, FPM = 0.3048, 0.514444, 0.3048 / 60
WGS84_A, WGS84_E2 = 6378137.0, 6.69437999014e-3
API = "https://api.github.com/repos/adsblol/globe_history_{y}/releases/tags/{tag}"


@dataclass
class Track:
    icao: str
    type: str          # ICAO type designator, e.g. "B738" ("" if unknown)
    reg: str
    t: np.ndarray      # unix seconds
    lat: np.ndarray
    lon: np.ndarray
    alt_m: np.ndarray  # metres; geometric (WGS84 ellipsoid) where available, else barometric shifted onto it
    alt_kind: np.ndarray  # "geom" | "baro_shifted" (onto GPS) | "baro" (no GPS in the track) | "none" | "ground"
    gs: np.ndarray     # ground speed, m/s
    track: np.ndarray  # degrees from north
    vrate: np.ndarray  # m/s
    leg: np.ndarray    # True where readsb marks a new leg or a stale position


def parse_readsb_trace(d):
    """readsb trace_full JSON -> Track. Point layout (readsb README-json): [dt, lat, lon,
    alt_baro_ft | "ground" | null, gs_kt, track_deg, flags, baro_rate_fpm, aircraft|null,
    source, alt_geom_ft, geom_rate_fpm, ...]."""
    rows = [p for p in d.get("trace", []) if p[1] is not None and p[2] is not None]
    n = len(rows)
    col = lambda i: [p[i] if len(p) > i else None for p in rows]
    num = lambda v: np.array([np.nan if x is None or isinstance(x, str) else float(x) for x in v])
    baro, geom = col(3), num(col(10))
    ground = np.array([b == "ground" for b in baro])
    baro = num(baro)
    both = np.isfinite(geom) & np.isfinite(baro)
    # barometric altitude reads tens of metres off GPS (pressure setting, and it drifts along a
    # flight): shift it by the GPS-baro difference interpolated in time from the nearest points
    # that report both, or a track switching between the two jumps up and down. With no such
    # point the barometric value stays as it is ("baro"; "baro_shifted" otherwise).
    t_all = float(d["timestamp"]) + num(col(0))
    off = np.interp(t_all, t_all[both], geom[both] - baro[both]) if both.any() else np.zeros(n)
    alt = np.where(np.isfinite(geom), geom, baro + off) * FT
    kind = np.where(ground, "ground", np.where(np.isfinite(geom), "geom", np.where(
        ~np.isfinite(baro), "none", "baro_shifted" if both.any() else "baro")))
    flags = np.array([int(f or 0) for f in col(6)], dtype=int)
    rate = num(col(11))
    rate = np.where(np.isfinite(rate), rate, num(col(7)))
    return Track(str(d.get("icao", "")).lower(), str(d.get("t", "") or ""), str(d.get("r", "") or ""),
                 float(d["timestamp"]) + num(col(0)), num(col(1)), num(col(2)), alt, kind,
                 num(col(4)) * KT, num(col(5)), rate * FPM, (flags & 3) > 0 if n else np.zeros(0, bool))


def to_enu(lat, lon, h, lat0, lon0, h0):
    """WGS84 geodetic -> local east/north/up metres around (lat0, lon0, h0)."""
    def ecef(la, lo, hh):
        la, lo = np.radians(la), np.radians(lo)
        n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(la) ** 2)
        return np.stack([(n + hh) * np.cos(la) * np.cos(lo), (n + hh) * np.cos(la) * np.sin(lo),
                         (n * (1 - WGS84_E2) + hh) * np.sin(la)], axis=-1)
    d = ecef(lat, lon, h) - ecef(lat0, lon0, h0)
    la, lo = np.radians(lat0), np.radians(lon0)
    R = np.array([[-np.sin(lo), np.cos(lo), 0],
                  [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                  [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]])
    return d @ R.T


def _ecef(la, lo, hh):
    la, lo = np.radians(la), np.radians(lo)
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(la) ** 2)
    return np.stack([(n + hh) * np.cos(la) * np.cos(lo), (n + hh) * np.cos(la) * np.sin(lo),
                     (n * (1 - WGS84_E2) + hh) * np.sin(la)], axis=-1)


def from_enu(e, n, u, lat0, lon0, h0):
    """Inverse of to_enu: local east/north/up metres around (lat0, lon0, h0) -> WGS84 (lat, lon, h)."""
    la0, lo0 = np.radians(lat0), np.radians(lon0)
    R = np.array([[-np.sin(lo0), np.cos(lo0), 0],
                  [-np.sin(la0) * np.cos(lo0), -np.sin(la0) * np.sin(lo0), np.cos(la0)],
                  [np.cos(la0) * np.cos(lo0), np.cos(la0) * np.sin(lo0), np.sin(la0)]])
    x, y, z = np.moveaxis(np.stack([e, n, u], axis=-1) @ R + _ecef(lat0, lon0, h0), -1, 0)
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - WGS84_E2))
    for _ in range(5):  # converges to well below a millimetre
        nn = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - nn
        lat = np.arctan2(z, p * (1 - WGS84_E2 * nn / (nn + h)))
    nn = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    return np.degrees(lat), np.degrees(lon), p / np.cos(lat) - nn


def segments(tr, max_gap_s=30.0):
    """Index arrays of continuous pieces (split at gaps, stale positions and new legs)."""
    if len(tr.t) == 0:
        return []
    cut = np.nonzero((np.diff(tr.t) > max_gap_s) | tr.leg[1:])[0] + 1
    return [s for s in np.split(np.arange(len(tr.t)), cut) if len(s) >= 2]


def stretch_boxes(tr, idx, frame, t0, t1, margin_m=100.0, smooth_s=1.0):
    """(boxes (m, 4) [e0, e1, n0, n1], stretch start times (m,), end times (m,)) for one segment idx:
    local metres around frame = (lat, lon, h) that provably contain everything interpolate() can place
    the segment at during each stretch between consecutive samples within [t0, t1] (bracketing samples
    included). The bounds (see locate.track_boxes): the cubic Hermite bound
    |H - chord| <= 0.0963 |dp| + 4/27 h (|v0| + |v1|), the smoothing reach (4 smooth_s + 0.1 s) * max |H'|
    with max |H'| <= 1.5 |dp| / h + |v0| + |v1|, |v| the larger of the reported and finite-difference
    velocity, and margin_m for frame and sub-point approximations (metres over <= 20 km)."""
    empty = (np.zeros((0, 4)), np.zeros(0), np.zeros(0))
    j = np.flatnonzero((tr.t[idx] >= t0) & (tr.t[idx] <= t1))
    if not len(j):
        return empty
    idx = idx[max(j[0] - 1, 0):j[-1] + 2]  # bracketing samples too
    idx = idx[np.concatenate([[True], np.diff(tr.t[idx]) > 1e-3])]  # as interpolate
    if len(idx) < 2:
        return empty
    t = tr.t[idx]
    z = tr.alt_m[idx].astype(float)
    z = np.interp(t, t[np.isfinite(z)], z[np.isfinite(z)]) if np.isfinite(z).any() else np.zeros(len(t))
    P = np.column_stack([to_enu(tr.lat[idx], tr.lon[idx], np.full(len(idx), frame[2]), *frame)[:, :2], z])
    fd = np.gradient(P, t, axis=0)
    v = np.hypot(np.nan_to_num(np.fmax(np.abs(tr.gs[idx]), np.hypot(fd[:, 0], fd[:, 1]))),
                 np.nan_to_num(np.fmax(np.abs(tr.vrate[idx]), np.abs(fd[:, 2]))))
    dp, h = np.linalg.norm(np.diff(P, axis=0), axis=1), np.diff(t)
    dev = 0.0963 * dp + 4 / 27 * h * (v[:-1] + v[1:])
    pad = dev + (4 * smooth_s + 0.1) * np.max(1.5 * dp / h + v[:-1] + v[1:]) + margin_m
    e, n = P[:, 0], P[:, 1]
    return (np.column_stack([np.minimum(e[:-1], e[1:]) - pad, np.maximum(e[:-1], e[1:]) + pad,
                             np.minimum(n[:-1], n[1:]) - pad, np.maximum(n[:-1], n[1:]) + pad]), t[:-1], t[1:])


def interpolate(tr, idx, t_new, mic, ground_h, smooth_s=1.0):
    """ENU positions (len(t_new), 3) along one segment: cubic Hermite through the reported
    positions using reported airborne velocities (finite differences on the ground, where
    readsb reports a quantised heading, and where missing), then a smooth_s Gaussian to
    remove position quantisation, which the Doppler ratio (a derivative) would amplify.
    Aircraft on the ground get altitude ground_h. No extrapolation: NaN outside."""
    t = tr.t[idx]
    keep = np.concatenate([[True], np.diff(t) > 1e-3])
    idx, t = idx[keep], t[keep]
    out = np.full((len(t_new), 3), np.nan)
    h = np.where(tr.alt_kind[idx] == "ground", ground_h, tr.alt_m[idx])
    if not np.isfinite(h).any():
        return out  # no altitude anywhere in this segment: can't place the aircraft
    if not np.all(np.isfinite(h)):
        h = np.interp(t, t[np.isfinite(h)], h[np.isfinite(h)])
    p = to_enu(tr.lat[idx], tr.lon[idx], h, *mic)
    v = np.stack([tr.gs[idx] * np.sin(np.radians(tr.track[idx])),
                  tr.gs[idx] * np.cos(np.radians(tr.track[idx])), tr.vrate[idx]], axis=1)
    fd = np.gradient(p, t, axis=0) if len(t) > 1 else np.zeros_like(p)
    v[tr.alt_kind[idx] == "ground"] = np.nan
    v = np.where(np.isfinite(v), v, fd)
    ok = (t_new >= t[0]) & (t_new <= t[-1])
    if ok.sum() >= 2 and len(t) >= 2:
        q = CubicHermiteSpline(t, p, v, axis=0)(t_new[ok])
        step = np.median(np.diff(t_new[ok]))
        if smooth_s and len(q) > 1:
            # pad by point reflection: a constant ('nearest') pad halves the speed at the ends
            n = min(int(4 * smooth_s / step) + 1, len(q) - 1)
            qp = np.concatenate([2 * q[:1] - q[n:0:-1], q, 2 * q[-1:] - q[-2:-n - 2:-1]])
            q = gaussian_filter1d(qp, smooth_s / step, axis=0, mode="nearest")[n:n + len(q)]
        out[ok] = q
    return out


class _Parts(io.RawIOBase):
    """The archive's .tar.aa/.ab/... parts as one sequential stream."""

    def __init__(self, urls, progress):
        self.urls, self.resp, self.read_total, self.progress = list(urls), None, 0, progress

    def readable(self):
        return True

    def readinto(self, b):
        while True:
            if self.resp is None:
                if not self.urls:
                    return 0
                self.resp = urllib.request.urlopen(self.urls.pop(0), timeout=120)
            n = self.resp.readinto(b)
            if n:
                before = self.read_total
                self.read_total += n
                if self.progress and self.read_total // 200_000_000 > before // 200_000_000:
                    self.progress(self.read_total)
                return n
            self.resp.close()
            self.resp = None


def _part_urls(day):
    tag = f"v{day.replace('-', '.')}-planes-readsb-prod-0"
    y = day[:4]
    with urllib.request.urlopen(API.format(y=y, tag=tag), timeout=60) as r:
        names = sorted(a["name"] for a in json.load(r)["assets"] if ".tar" in a["name"])
    return [f"https://github.com/adsblol/globe_history_{y}/releases/download/{tag}/{n}" for n in names]


def fetch_adsblol(day, lat0, lon0, radius_km=20.0, cache_dir="data/adsb", progress=print):
    """All aircraft traces of UTC `day` ("YYYY-MM-DD") that came within radius_km of
    (lat0, lon0) at any time that day. Streams the archive once; cached afterwards."""
    return fetch_adsblol_sites(day, [(lat0, lon0)], radius_km, cache_dir, progress)[0]


def fetch_adsblol_sites(day, sites, radius_km=20.0, cache_dir="data/adsb", progress=print):
    """fetch_adsblol for several (lat, lon) sites at once: one ~2-4 GB stream of the day's
    archive serves every site that isn't cached yet. Returns one list of Tracks per site."""
    caches = [Path(cache_dir) / f"adsblol_{day}_{la:.3f}_{lo:.3f}_{radius_km:g}km.json.gz" for la, lo in sites]
    todo = [i for i, c in enumerate(caches) if not c.exists()]
    if todo:
        boxes = [(sites[i][0], sites[i][1], radius_km / 111.0,
                  radius_km / (111.0 * max(np.cos(np.radians(sites[i][0])), 0.05))) for i in todo]
        kept = [[] for _ in todo]
        stream = io.BufferedReader(_Parts(_part_urls(day), lambda n: progress(
            f"  {day}: streamed {n / 1e9:.1f} GB, kept {sum(map(len, kept))} traces")), 1 << 20)
        with tarfile.open(fileobj=stream, mode="r|") as tf:
            for m in tf:
                if not (m.isfile() and "trace_full_" in m.name):
                    continue
                raw = tf.extractfile(m).read()
                d = json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
                pts = np.array([(p[1], p[2]) for p in d.get("trace", []) if p[1] is not None and p[2] is not None])
                if not len(pts):
                    continue
                for k, (la, lo, dla, dlo) in enumerate(boxes):
                    if np.any((np.abs(pts[:, 0] - la) < dla) & (np.abs(pts[:, 1] - lo) < dlo)):
                        kept[k].append(d)
        for k, i in enumerate(todo):
            caches[i].parent.mkdir(parents=True, exist_ok=True)
            caches[i].write_bytes(gzip.compress(json.dumps(kept[k]).encode()))
            progress(f"  {day}: {len(kept[k])} aircraft near {sites[i]}, cached in {caches[i]}")
    return [[parse_readsb_trace(d) for d in json.loads(gzip.decompress(c.read_bytes()))] for c in caches]


def in_box(d, box):
    """Does a readsb trace have a position inside box = (lat_min, lat_max, lon_min, lon_max)?"""
    pts = np.array([(p[1], p[2]) for p in d.get("trace", []) if p[1] is not None and p[2] is not None])
    return bool(len(pts)) and bool(np.any((pts[:, 0] >= box[0]) & (pts[:, 0] <= box[1]) & (pts[:, 1] >= box[2]) & (pts[:, 1] <= box[3])))


def stream_day(day, boxes, sink, progress=print):
    """Stream one day's adsb.lol archive once (3-4 GB, nothing written): every trace touching box k
    (lat_min, lat_max, lon_min, lon_max) of boxes {k: box} goes to sink(k, trace dict)."""
    n = dict.fromkeys(boxes, 0)
    stream = io.BufferedReader(_Parts(_part_urls(day), lambda r: progress(f"  {day}: streamed {r / 1e9:.1f} GB, kept {n}")), 1 << 20)
    with tarfile.open(fileobj=stream, mode="r|") as tf:
        for m in tf:
            if not (m.isfile() and "trace_full_" in m.name):
                continue
            raw = tf.extractfile(m).read()
            d = json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
            for k, b in boxes.items():
                if in_box(d, b):
                    sink(k, d)
                    n[k] += 1
    return n


def stream_day_boxes(day, boxes, out_dir="data/adsb", progress=print):
    """Stream one day's adsb.lol archive once and write every trace touching each named box
    ({name: (lat_min, lat_max, lon_min, lon_max)}) to out_dir/adsblol_<day>_<name>.jsonl.gz as it
    goes (one JSON trace per line: memory stays small however large the box). Existing files are
    kept (delete them to redo). Returns {name: path}."""
    paths = {k: Path(out_dir) / f"adsblol_{day}_{k}.jsonl.gz" for k in boxes}
    todo = {k: b for k, b in boxes.items() if not paths[k].exists()}
    if todo:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        tmp = {k: paths[k].with_suffix(".tmp") for k in todo}
        outs = {k: gzip.open(tmp[k], "wt") for k in todo}
        n = stream_day(day, todo, lambda k, d: outs[k].write(json.dumps(d) + "\n"), progress)
        for k in todo:
            outs[k].close()
            tmp[k].replace(paths[k])  # only a complete stream becomes the cached file
            progress(f"  {day}: {n[k]} aircraft in box {k}, cached in {paths[k]}")
    return paths


def load_box(path, box=None, t0=None, t1=None):
    """Tracks from a stream_day_boxes file, optionally only those inside a smaller box and active
    between t0 and t1 (unix s)."""
    out = []
    with gzip.open(path, "rt") as f:
        for line in f:
            d = json.loads(line)
            if box is not None and not in_box(d, box):
                continue
            tr = parse_readsb_trace(d)
            if t0 is not None and (len(tr.t) == 0 or tr.t[-1] < t0 or tr.t[0] > t1):
                continue
            out.append(tr)
    return out
