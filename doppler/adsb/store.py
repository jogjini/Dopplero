"""Whole-day ADS-B store: one stream of a day's adsb.lol archive, kept for the whole world, already
parsed, so later searches read only the hours and area they need (owner's request 2026-11-07).

Layout: data/adsb/world/<day>/
  aircraft.json             index -> [icao, type, reg]
  <HH>_<part>.npz           points whose time falls in UTC hour HH, written in parts as the stream
                            goes (memory stays bounded): aircraft index (int32), t (float64, unix s),
                            lat, lon (float64, WGS84), alt_m (float32, above the WGS84 ellipsoid as
                            parse_readsb_trace gives it), alt_kind (uint8 code), gs, track, vrate
                            (float32), leg (bool)
  complete                  written last: a store without it is unfinished and is not used
Lossless with respect to parse_readsb_trace (float32 for altitude, speeds and angles: < 1 mm, < 1 mm/s).
"""
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from .tracks import Track, parse_readsb_trace, stream_day

ROOT = Path("data/adsb/world")
KINDS = ["geom", "baro_shifted", "baro", "none", "ground"]
FIELDS = ("t", "lat", "lon", "alt_m", "gs", "track", "vrate")
FLUSH_POINTS = 15_000_000


def day_dir(day, root=ROOT):
    return Path(root) / day


def is_stored(day, root=ROOT):
    return (day_dir(day, root) / "complete").exists()


class _Writer:
    def __init__(self, d):
        self.d, self.ac, self.idx, self.buf, self.n, self.parts = d, [], {}, defaultdict(list), 0, defaultdict(int)

    def add(self, tr):
        if not len(tr.t):
            return
        key = (tr.icao, tr.type, tr.reg)
        a = self.idx.setdefault(key, len(self.ac))
        if a == len(self.ac):
            self.ac.append(list(key))
        hours = np.floor((tr.t % 86400) / 3600).astype(int)
        kind = np.array([KINDS.index(k) if k in KINDS else KINDS.index("none") for k in tr.alt_kind], np.uint8)
        for h in np.unique(hours):
            m = hours == h
            self.buf[int(h)].append((np.full(m.sum(), a, np.int32), tr.t[m], tr.lat[m], tr.lon[m], tr.alt_m[m], tr.gs[m],
                                     tr.track[m], tr.vrate[m], kind[m], tr.leg[m]))
        self.n += len(tr.t)
        if self.n >= FLUSH_POINTS:
            self.flush()

    def flush(self):
        for h, rows in self.buf.items():
            cols = [np.concatenate(c) for c in zip(*rows)]
            np.savez_compressed(self.d / f"{h:02d}_{self.parts[h]:03d}.npz", aircraft=cols[0], t=cols[1], lat=cols[2], lon=cols[3],
                                alt_m=cols[4].astype(np.float32), gs=cols[5].astype(np.float32), track=cols[6].astype(np.float32),
                                vrate=cols[7].astype(np.float32), alt_kind=cols[8], leg=cols[9])
            self.parts[h] += 1
        self.buf, self.n = defaultdict(list), 0

    def close(self):
        self.flush()
        (self.d / "aircraft.json").write_text(json.dumps(self.ac))
        (self.d / "complete").write_text("ok\n")


def store_day(day, root=ROOT, progress=print, source=None):
    """Stream `day` once (whole world) into the store; returns its directory. source: an iterable of
    readsb trace dicts instead of the network stream (tests)."""
    d = day_dir(day, root)
    if is_stored(day, root):
        return d
    if Path(root).is_symlink() and not Path(root).exists():  # e.g. the store's external disk isn't connected
        raise RuntimeError(f"the flight store {root} links to {os.readlink(root)}, which isn't there: connect that disk, "
                           "or untick 'keep flight data on disk' to stream the day instead")
    if d.exists():
        shutil.rmtree(d)  # an unfinished earlier attempt
    d.mkdir(parents=True)
    w = _Writer(d)
    if source is None:
        stream_day(day, {"world": (-90.0, 90.0, -180.0, 180.0)}, lambda k, tr: w.add(parse_readsb_trace(tr)), progress)
    else:
        for tr in source:
            w.add(parse_readsb_trace(tr))
    w.close()
    progress(f"  {day}: stored {len(w.ac)} aircraft in {d}")
    return d


def days(t0, t1):
    """The UTC days a time window (unix s) touches."""
    d0, d1 = (np.datetime64(int(t), "s").astype("datetime64[D]") for t in (t0, t1))
    return [str(d) for d in np.arange(d0, d1 + 1)]


def load(box, t0, t1, root=ROOT, pad_s=120.0):
    """Tracks of every aircraft with a point inside box = (lat_min, lat_max, lon_min, lon_max) during
    [t0, t1] (unix s): each with ALL its points in [t0 - pad_s, t1 + pad_s], inside the box or not and
    from every day the window touches (one Track per aircraft, joined across midnight, time-sorted),
    so interpolation near the box edges and window ends sees the same samples as a whole-track load.
    Every day of [t0, t1] must be stored; a day only the padding reaches is used if stored, else its few
    minutes are left out (a window starting at 00:00 must not need the day before)."""
    w0, w1 = t0 - pad_s, t1 + pad_s
    meta, key, parts = [], {}, []
    for day in days(w0, w1):
        d = day_dir(day, root)
        if not is_stored(day, root):
            if day in days(t0, t1):
                raise FileNotFoundError(f"{day} is not stored in {root}")
            continue
        ac = json.loads((d / "aircraft.json").read_text())
        lut = np.empty(len(ac), np.int64)  # this day's aircraft index -> one index per icao over all days
        for i, a in enumerate(ac):
            if a[0] not in key:
                key[a[0]] = len(meta)
                meta.append(a)
            lut[i] = key[a[0]]
        day0 = float(np.datetime64(day).astype("datetime64[s]").astype(np.int64))
        h0, h1 = int(max(0, (w0 - day0) // 3600)), int(min(23, (w1 - day0) // 3600))
        for h in range(h0, h1 + 1):
            for f in sorted(d.glob(f"{h:02d}_*.npz")):
                z = np.load(f)
                m = (z["t"] >= w0) & (z["t"] <= w1)
                if m.any():
                    q = {k: z[k][m] for k in ("aircraft", "alt_kind", "leg") + FIELDS}
                    q["aircraft"] = lut[q["aircraft"]]
                    parts.append(q)
    if not parts:
        return []
    c = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    inside = (c["t"] >= t0) & (c["t"] <= t1) & (c["lat"] >= box[0]) & (c["lat"] <= box[1]) & (c["lon"] >= box[2]) & (c["lon"] <= box[3])
    keep = np.isin(c["aircraft"], np.unique(c["aircraft"][inside]))
    c = {k: v[keep] for k, v in c.items()}
    if not len(c["t"]):
        return []
    order = np.lexsort((c["t"], c["aircraft"]))
    c = {k: v[order] for k, v in c.items()}
    cuts = np.flatnonzero(np.diff(c["aircraft"])) + 1
    out = []
    kinds = np.array(KINDS)
    for idx in np.split(np.arange(len(order)), cuts):
        icao, typ, reg = meta[int(c["aircraft"][idx[0]])]
        out.append(Track(icao, typ, reg, c["t"][idx], c["lat"][idx], c["lon"][idx], c["alt_m"][idx].astype(float),
                         kinds[c["alt_kind"][idx]], c["gs"][idx].astype(float), c["track"][idx].astype(float),
                         c["vrate"][idx].astype(float), c["leg"][idx]))
    return out


if __name__ == "__main__":
    import sys
    for day in sys.argv[1:]:  # uv run python -m doppler.adsb.store 2026-09-08 [more days]
        store_day(day, progress=lambda m: print(m, flush=True))
