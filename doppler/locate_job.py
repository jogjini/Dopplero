"""One localisation job (owner's request 2026-11-06/07; design review 038): audio + time + search box
-> ranked candidate places, with everything measured kept in the job folder for later analysis.

Use only on recordings you have the right to locate (public recordings with published locations
for testing).

Steps (each written to the job folder as it finishes):
  1. audio: decoded, sha256                                     -> audio.json
  2. the spectrogram strips of every 15-min chunk; the curves are the ones marked by hand
     (manual.json, read by manual_curves)                       -> detections.json
  3. flights: the UTC day(s) covering the recording, read from the whole-world day store
     (data/adsb/world; disk mode stores a missing day first) or streamed with only the box kept (RAM)
                                                                -> flights.json (summary per aircraft)
  4. candidates: fast path (trusted, not car-like fits x flights of matching speed, with a wind
     allowance: Doppler measures airspeed, ADS-B ground speed), optional joint proposals (two trusted
     fits on two aircraft meeting at one place with one shared clock offset; off by default: no app
     job ever got one, and they grow as fits x flights: a 12 h, 1,000 km job ran out of memory),
     plus an optional grid
  5. scores: every candidate scored with all detections (locate.score_place, one shared clock);
     the same in decoy skies: the flights of +-null_shift_s (0 = automatic: >= 15 min and beyond the
     recording + clock window), optionally another day's  -> candidates.json, nulls.json
  6. result: ranking; best real vs best fake sky (information only: no calibrated claim yet)
                                                                -> result.json
All progress lines -> progress.log; parameters -> params.json; per-step runtimes in result.json.
"""
import functools
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import audio_io, evidence, locate, physics, search, trackmatch
from . import spectrogram as sg
from .adsb import store
from .adsb.tracks import parse_readsb_trace, stream_day

CHUNK_S, MARGIN_S = 900.0, 180.0   # as scripts/label_eclipse.py (the frozen chunking)
DEFAULTS = {"start_utc": None, "clock_s": 15.0, "box": None, "data_mode": "ram", "workers": os.cpu_count(),
            "wind_mps": 25.0, "temperature_c": 15.0, "mic_height_m": 1.5, "ground_elev_m": "auto",
            "min_score": 15.0, "car_speed_mps": 25.0, "car_distance_m": 150.0, "grid_m": 0.0,
            "fake_sky": True, "null_shift_s": 0.0, "other_day_null": 0, "silence_penalty": 0.0, "refine": False, "joint": False, "detect_only": False, "detector": "manual", "search": "fits", "flying_only": True, "car_filter": False, "claim_margin": 0.25, "margin_km": 15.0, "top_n": 20}


SPEC_FMIN, SPEC_FMAX = 30.0, 3000.0   # the UI's full-band strip (log frequency) and pixels per second
LIN_FMIN, LIN_FMAX, LIN_ROWS = 30.0, 600.0, 571  # the UI's propeller band (linear frequency, 1 row per Hz)
SPEC_PX_PER_S = 2
SPEC_DB = (-3.0, 18.0)  # colour range in whitened dB (noise floor ~0; strong tones 12-20)


def _strip(job_dir, g0, W1, W4, dur):
    """The chunk's core (900 s) as two colour PNGs for the UI (whitened dB, broadband sound kept, averaged
    over each pixel's time): full band, log frequency, 1 s window (W1); 30-600 Hz, linear frequency, 4 s
    window (W4: steady tones sharpest, the clearest view of propeller harmonics). Plus what the UI needs
    to draw curves on them. Display only."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    lo, hi = int(sg.freq_to_bin(SPEC_FMIN)), int(sg.freq_to_bin(SPEC_FMAX))
    c0 = int(round(MARGIN_S / sg.HOP_S))
    c1 = min(int(round((MARGIN_S + CHUNK_S) / sg.HOP_S)), W1.shape[1], int(round((dur - g0) / sg.HOP_S)))
    if c1 <= c0:
        return None
    A, A4 = (np.nan_to_num(X[:, c0:c1], nan=SPEC_DB[0]) for X in (W1, W4))
    step = max(1, int(round(1 / (sg.HOP_S * SPEC_PX_PER_S))))  # columns -> SPEC_PX_PER_S px per second
    pool = lambda img: img[:, : (img.shape[1] // step) * step].reshape(img.shape[0], -1, step).mean(axis=2)
    full = pool(A[lo:hi])
    if full.shape[1] == 0:  # the last chunk can hold only a few frames of audio
        return None
    r = sg.freq_to_bin(np.linspace(LIN_FMIN, LIN_FMAX, LIN_ROWS))  # fractional rows of the log axis
    k = np.floor(r).astype(int)
    w = (r - k)[:, None]
    lin = pool(A4[k] * (1 - w) + A4[k + 1] * w)
    name = f"spec_{g0 + MARGIN_S:07.0f}"
    for img, n in ((full, f"{name}.png"), (lin, f"{name}_p.png")):
        plt.imsave(Path(job_dir) / n, img[::-1], cmap="inferno", vmin=SPEC_DB[0], vmax=SPEC_DB[1])
    return {"file": f"{name}.png", "file_lin": f"{name}_p.png", "t0_s": g0 + MARGIN_S, "t1_s": g0 + c1 * sg.HOP_S,
            "bin_lo": lo, "bin_hi": hi, "fmin_hz": float(sg.FMIN), "bins_per_octave": int(sg.BPO), "lin_hz": [LIN_FMIN, LIN_FMAX]}


RENDER_PAD_S = 60.0  # audio read either side of a rendered span, so whitening (~30 s median) sees full context
RENDER_CACHE, RENDER_KEEP = Path("data/cache/render"), 3  # recordings decoded for the viewer, the latest few kept
_RAW_LOCK = threading.Lock()


def raw_audio(path):
    """The recording at path decoded once (float32, audio_io.SR) into the viewer's cache and memory-mapped: each
    render then reads its slice instead of running ffmpeg again (the decode was ~60 % of a render's time)."""
    s = Path(path).resolve().stat()
    key = hashlib.sha1(f"{Path(path).resolve()}|{s.st_size}|{s.st_mtime_ns}".encode()).hexdigest()[:16]
    f = RENDER_CACHE / f"{key}.f32"
    with _RAW_LOCK:  # one decode at a time: two renders of a new recording wait for the same file
        if not f.exists():
            RENDER_CACHE.mkdir(parents=True, exist_ok=True)
            tmp = f.with_suffix(".part")
            audio_io.to_raw(path, tmp)
            tmp.rename(f)
            for old in sorted(RENDER_CACHE.glob("*.f32"), key=lambda q: q.stat().st_mtime)[:-RENDER_KEEP]:
                old.unlink(missing_ok=True)
        f.touch()  # most recently used: kept longest
    return np.memmap(f, np.float32, mode="r") if f.stat().st_size else np.zeros(0, np.float32)


def _linear_db(x, sr, win_s, fmax):
    """Whitened dB on a LINEAR frequency grid (rows 0..fmax at 1/win_s Hz, frames at sg.HOP_S centred on
    i * HOP_S): scipy's Hann STFT, finer than the detector's log grid at high frequencies. The audio is
    first decimated as far as fmax allows."""
    from scipy import signal
    q = max(1, min(8, int(sr / (2.5 * fmax))))
    if q > 1:
        x, sr = signal.decimate(x, q, ftype="fir", zero_phase=True), sr // q
    n = int(round(win_s * sr))
    stft = signal.ShortTimeFFT(signal.windows.hann(n), hop=int(round(sg.HOP_S * sr)), fs=sr, scale_to="psd")
    P = stft.spectrogram(x, p0=0, p1=len(x) // stft.hop + 1)  # slice i centred on sample i * hop
    return sg.whiten_db(P[: int(fmax * n / sr) + 2].astype(np.float32)), sr / n  # (rows, frames) dB, Hz per row


def render(path, t0, t1, fmin, fmax, axis="lin", win_s=1.0, px_per_s=2, height=330):
    """PNG bytes of the recording at path over [t0, t1] s: whitened dB from a linear-frequency STFT with a
    win_s Hann window (_linear_db; sharper than the detector's log grid above ~200 Hz), on fmin..fmax Hz,
    one row per pixel on a linear or log ("log") axis, averaged to px_per_s columns per second (any value up to 8,
    the analysis' frame rate); columns
    past the end of the audio are the noise-floor colour. The UI's adjustable view (display only)."""
    import io
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    a = max(0.0, t0 - RENDER_PAD_S)
    x = np.asarray(raw_audio(path)[int(a * audio_io.SR):int((t1 + RENDER_PAD_S) * audio_io.SR)])
    with np.errstate(divide="ignore", invalid="ignore"):
        C, df = _linear_db(x, audio_io.SR, win_s, fmax)
    C = np.nan_to_num(C, nan=0.0)
    i0 = int(round((t0 - a) / sg.HOP_S))
    n_fr = max(1, int(round((t1 - t0) / sg.HOP_S)))
    n_px = max(1, min(n_fr, int(round((t1 - t0) * px_per_s))))  # any zoom; at most one column per frame (8 per second)
    C = C[:, i0:i0 + n_fr]
    f = np.geomspace(fmin, fmax, height) if axis == "log" else np.linspace(fmin, fmax, height)
    r = np.clip(f / df, 0, C.shape[0] - 1.001)
    k = np.floor(r).astype(int)
    w = (r - k)[:, None]
    img = C[k] * (1 - w) + C[k + 1] * w
    img = np.hstack([img, np.full((height, n_fr - img.shape[1]), SPEC_DB[0])])  # past the end of the audio
    edges = np.linspace(0, n_fr, n_px + 1).astype(int)  # the frames spread evenly over the columns, each averaged
    img = np.add.reduceat(img, edges[:-1], axis=1) / np.diff(edges)
    buf = io.BytesIO()
    plt.imsave(buf, img[::-1], cmap="inferno", vmin=SPEC_DB[0], vmax=SPEC_DB[1], format="png")
    return buf.getvalue()


def _chunk(job):
    """One chunk of the audio's UI strip (its curves are marked by hand: no detections)."""
    path, g0, job_dir = job
    x = np.memmap(path, np.float32, mode="r")  # the decoded recording; only this chunk's slice is read
    i0, i1 = int(g0 * audio_io.SR), int((g0 + CHUNK_S + 2 * MARGIN_S) * audio_io.SR)
    seg, valid = np.zeros(i1 - i0, np.float32), np.zeros(i1 - i0, bool)
    a, b = max(i0, 0), min(i1, len(x))
    seg[a - i0:b - i0] = x[a:b]
    valid[a - i0:b - i0] = np.isfinite(seg[a - i0:b - i0])  # NaN = a gap between files
    seg[~valid] = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        F = sg.features(seg)
    ok = _frame_valid(valid, F[1].shape[1])
    strip = _strip(job_dir, g0, np.where(ok, F[1], np.nan), np.where(ok, F[2], np.nan), len(x) / audio_io.SR)
    return [], strip  # no automatic detection: the curves are marked by hand (manual_curves)


MANUAL_ID0 = 100000  # hand-fitted curves are numbered from here (above any detection of older jobs)
# shortcut: one hand fit has no measured spread, so a pass's ranges are fixed fractions; derive them from the
# fit's own sensitivity if hand fits are ever scored against the automatic ones
MANUAL_SPREAD = {"speed": (0.9, 1.1), "distance": (0.75, 1.33), "time_s": 1.0}


def manual_curves(job_dir):
    """The job's curves: its detections plus the curves the user fitted by hand (manual.json, written by the
    desktop app's curve tool) as detections: numbered after every detection (from MANUAL_ID0), one group each,
    their saved lines as contours; a straight-pass model also gives a fit, trusted since the user matched it."""
    job_dir = Path(job_dir)
    dets = json.loads((job_dir / "detections.json").read_text())
    f = job_dir / "manual.json"
    if not f.exists():
        return dets
    first = max([MANUAL_ID0] + [int(d["id"]) + 1 for d in dets])
    out = []
    for k, cv in enumerate(json.loads(f.read_text())):
        cs = [{"t_s": q["t_s"], "f_hz": q["f_hz"]} for q in cv["contours"] if len(q["t_s"]) >= 2]
        if not cs:
            continue
        ps, (lo_v, hi_v), (lo_d, hi_d) = cv.get("pass"), MANUAL_SPREAD["speed"], MANUAL_SPREAD["distance"]
        rng = lambda v, lo, hi: {"median": v, "p05": v * lo, "p95": v * hi}
        fit_ = ps and {"speed_mps": rng(ps["speed_mps"], lo_v, hi_v), "cpa_distance_m": rng(ps["distance_m"], lo_d, hi_d),
                       "cpa_time_s": {"median": ps["t_cpa_s"], "p05": ps["t_cpa_s"] - MANUAL_SPREAD["time_s"],
                                      "p95": ps["t_cpa_s"] + MANUAL_SPREAD["time_s"]},
                       "flags": ["hand fit"], "geometry_supported": True, "residual_cents": cv.get("residual_cents") or 0.0}
        out.append({"id": first + k, "t_cpa_s": ps["t_cpa_s"] if ps else float(np.mean(cv["extent"])), "score": 0.0,
                    "f0_hz": ps["f0_hz"] if ps else float(np.median(cs[0]["f_hz"])), "group": f"manual:{k}", "representative": True,
                    "manual": True, "model": cv["model"], "fit": fit_, "contours": cs,
                    "points": cv.get("points", []) + cv.get("traced", [])})  # what the user clicked: the track search's input
    return dets + out


BLOCK = 60 * audio_io.SR  # samples copied per step when assembling: memory stays bounded


def _decode(p, start, raw, job_dir):
    """Step 1: the recording decoded into raw (float32, audio_io.SR), as one file (audio_path) or as
    several timestamped files (audio_files: [{"path", "start_utc"}], e.g. an AudioMoth day) placed at
    their offsets from start: samples no file covers are NaN (invalid frames, as the labeller's gaps);
    overlaps keep the earlier-starting file's samples. Returns [{"file", "offset_s", "overlap_s",
    "sha256"}] in time order."""
    if not p.get("audio_files"):
        audio_io.to_raw(Path(p["audio_path"]), raw)
        files = [{"file": Path(p["audio_path"]).name, "path": p["audio_path"], "offset_s": 0.0, "overlap_s": 0.0}]
    else:
        when = lambda q: datetime.fromisoformat(q["start_utc"].replace("Z", "+00:00")).astimezone(timezone.utc).timestamp() - start
        order = sorted(p["audio_files"], key=when)
        assert when(order[0]) >= 0, "start_utc must not be after the first file's start"
        tmp, files = Path(job_dir) / "one.f32", []
        try:
            with open(raw, "wb") as fh:
                end = 0  # samples written so far
                for q in order:
                    audio_io.to_raw(Path(q["path"]), tmp)
                    y = np.memmap(tmp, np.float32, mode="r") if tmp.stat().st_size else np.zeros(0, np.float32)
                    i = int(round(when(q) * audio_io.SR))
                    for k in range(end, i, BLOCK):  # the gap up to this file
                        np.full(min(BLOCK, i - k), np.nan, np.float32).tofile(fh)
                    end = max(end, i)
                    skip = min(end - i, len(y))  # overlap with what is already written
                    for k in range(skip, len(y), BLOCK):
                        np.asarray(y[k:k + BLOCK]).tofile(fh)
                    end += len(y) - skip
                    del y
                    files.append({"file": Path(q["path"]).name, "path": q["path"], "offset_s": i / audio_io.SR, "overlap_s": skip / audio_io.SR})
        finally:
            tmp.unlink(missing_ok=True)
    for q in files:
        with open(q.pop("path"), "rb") as fh:
            q["sha256"] = hashlib.file_digest(fh, "sha256").hexdigest()
    return files


def _frame_valid(valid, n_frames):
    """Frames whose 1 s window lies entirely on recorded audio (scripts/label_eclipse.frame_valid)."""
    bad = np.concatenate([[0], np.cumsum(~valid)])
    c = np.rint(sg.frame_times(n_frames) * audio_io.SR).astype(int)
    lo, hi = np.clip(c - audio_io.SR // 2, 0, len(valid)), np.clip(c + audio_io.SR // 2, 0, len(valid))
    return bad[hi] - bad[lo] == 0


_W = {}
REFINE_N = 5


def _dist(a, b):
    return float(np.hypot(*locate.to_local(a["lat"], a["lon"], b["lat"], b["lon"])))


def flights(tracks, w0, w1, box):
    """The aircraft with a point inside box during [w0, w1], one track each with all its points in
    [w0, w1], time-sorted, whichever way the day(s) were loaded: a flight crossing midnight arrives as
    one piece per daily archive and is joined here."""
    by = {}
    for tr in tracks:
        by.setdefault(tr.icao, []).append(tr)
    out = []
    for trs in by.values():
        t = np.concatenate([q.t for q in trs])
        o = np.argsort(t, kind="stable")
        o = o[(t[o] >= w0) & (t[o] <= w1)]
        lat, lon = (np.concatenate([getattr(q, k) for q in trs])[o] for k in ("lat", "lon"))
        if len(o) >= 2 and np.any((lat >= box[0]) & (lat <= box[1]) & (lon >= box[2]) & (lon <= box[3])):
            out.append(type(trs[0])(**{k: (np.concatenate([getattr(q, k) for q in trs])[o] if isinstance(v, np.ndarray) and len(v) == len(trs[0].t) else v)
                                       for k, v in trs[0].__dict__.items()}))
    return out


def _init(dets, n, start, c, h_ground, h_mic, clock_s, miss, flying):
    locate.CLOCK_S = clock_s  # explicit: a spawned worker does not inherit the parent's module state
    _W.update(dets=dets, n=n, start=start, c=c, hg=h_ground, hm=h_mic, miss=miss, flying=flying)


def _score(task):
    """One batch of places and the tracks they can see (a tile's near() set, or all tracks without
    an index): workers never hold every flight (a 12 h recording in a 1,000 km box: ~1 GB each)."""
    pts, trs = task
    return [locate.score_place(_W["dets"], locate.predictions(trs, p["lat"], p["lon"], _W["hm"], _W["start"], _W["n"], _W["c"], _W["hg"]),
                               miss=_W["miss"], flying_only=_W["flying"]) for p in pts]


def _ground(box, elev):
    lat, lon = (box[0] + box[1]) / 2, (box[2] + box[3]) / 2
    get = lambda u: json.load(__import__("urllib.request").request.urlopen(u, timeout=30))
    try:  # NOAA GEOID18 (US only): heights above the WGS84 ellipsoid = elevation + geoid height
        geoid, src = float(get(f"https://geodesy.noaa.gov/api/geoid/ght?lat={lat}&lon={lon}")["geoidHeight"]), "NOAA GEOID18"
    except Exception:
        geoid, src = 0.0, "none (outside the US or offline): heights may be off by up to ~100 m"
    if elev == "auto":
        try:
            elev = float(get(f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}")["elevation"][0])
        except Exception:
            elev = 0.0
    return float(elev), geoid, src


def _load_flights(a, b, keep, lo, hi, off=0.0):
    """The flights of [a, b] (clipped to [lo, hi]) in the box keep, read from the day store; off: another sky's, moved
    back by off. Picklable (functools.partial), so a search worker can read its own."""
    a, b = max(a, lo), min(b, hi)
    if b <= a:
        return []
    trs = flights(store.load(keep, a + off, b + off), a + off, b + off, keep)
    return [type(t)(**{**t.__dict__, "t": t.t - off}) for t in trs] if off else trs


POLISH_KM, POLISH_CLOCK_S = 3.0, 5.0  # the curve-shape search's last step: a small box around its best place
WEATHER_LEVELS = ("10m", "925hPa", "850hPa")  # ground to ~1.5 km: where the sound of a light plane travels


def _weather(lat, lon, t0, t1):
    """(mean air temperature in C, mean wind (east, north) in m/s) from the ground to ~1.5 km at (lat, lon) over the
    hours [t0, t1] (unix), from Open-Meteo's historical forecasts (free, keyless); None when it can't be had (no
    network, a date it doesn't cover): the search then keeps its defaults. The wind is the direction the air moves to
    (the reports give where it comes from)."""
    d0, d1 = (datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d") for t in (t0, t1))
    url = (f"https://historical-forecast-api.open-meteo.com/v1/forecast?latitude={lat:.4f}&longitude={lon:.4f}"
           f"&start_date={d0}&end_date={d1}&wind_speed_unit=ms&timezone=UTC&hourly="
           + ",".join([f"temperature_{lv}" for lv in ("2m", "925hPa", "850hPa")]
                      + [f"wind_{v}_{lv}" for lv in WEATHER_LEVELS for v in ("speed", "direction")]))
    try:
        h = json.load(__import__("urllib.request").request.urlopen(url, timeout=30))["hourly"]
    except Exception:
        return None
    hours = [i for i, t in enumerate(h["time"]) if t0 - 3600 < datetime.fromisoformat(t + "+00:00").timestamp() < t1 + 3600]
    temps = [h[f"temperature_{lv}"][i] for lv in ("2m", "925hPa", "850hPa") for i in hours if h[f"temperature_{lv}"][i] is not None]
    winds = [(h[f"wind_speed_{lv}"][i], h[f"wind_direction_{lv}"][i]) for lv in WEATHER_LEVELS for i in hours
             if h[f"wind_speed_{lv}"][i] is not None and h[f"wind_direction_{lv}"][i] is not None]
    if not temps or not winds:
        return None
    to = lambda sp, dr: (-sp * np.sin(np.radians(dr)), -sp * np.cos(np.radians(dr)))  # from-direction -> motion
    we, wn = np.mean([to(sp, dr) for sp, dr in winds], axis=0)
    return float(np.mean(temps)), (float(we), float(wn))


def run(job_dir, params, log=print):
    job_dir = Path(job_dir)
    p = {**DEFAULTS, **params}
    (job_dir / "params.json").write_text(json.dumps(p, indent=1))
    plog = open(job_dir / "progress.log", "a")

    def say(msg):
        line = f"{datetime.now():%H:%M:%S} {msg}"
        plog.write(line + "\n")
        plog.flush()
        log(line)

    times, T = {}, time.time()
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = None
    box = None if p["detect_only"] else [float(v) for v in p["box"]]
    # detect_only needs no start time (times are then seconds into the file)
    start = datetime.fromisoformat(p["start_utc"].replace("Z", "+00:00")).astimezone(timezone.utc).timestamp() if p.get("start_utc") else 0.0
    c = float(physics.speed_of_sound(float(p["temperature_c"])))
    locate.CLOCK_S = float(p["clock_s"])
    workers = max(1, int(p["workers"]))

    # 1. audio
    raw = job_dir / "audio.f32"  # decoded once to disk: hours-long recordings don't fit in RAM once per worker
    files = _decode(p, start, raw, job_dir)
    name = files[0]["file"] if len(files) == 1 else f"{len(files)} files"
    n_samples = raw.stat().st_size // 4
    dur = n_samples / audio_io.SR
    (job_dir / "audio.json").write_text(json.dumps({**(files[0] if len(files) == 1 else {"files": files}),
                                                    "duration_s": dur, "start_utc": p["start_utc"], "commit": commit}, indent=1))
    say(f"audio {name}: {dur / 60:.1f} min; speed of sound {c:.1f} m/s")
    day0 = start
    if p.get("date_only") and p.get("start_utc"):  # the date only: any start that UTC day (start_utc = the date's 00:00)
        half = max(0.0, (86400.0 - dur) / 2)
        start, p["clock_s"] = day0 + half, half
        locate.CLOCK_S = half
        say(f"date only: every start from 00:00 to {datetime.fromtimestamp(day0 + 2 * half, timezone.utc):%H:%M} UTC is searched")
    times["audio"] = time.time() - T

    # 2. our pipeline
    T = time.time()
    chunks = list(np.arange(-MARGIN_S, dur, CHUNK_S))
    if not p.get("detections_from"):
        say(f"detecting in {len(chunks)} chunk(s) on {min(workers, len(chunks))} worker(s)")
    dets, strips = [], []
    try:
        if p.get("detections_from"):  # curves the user reviewed: a finished job's detections (same ids), detection skipped
            src = Path(p["detections_from"])
            keep_ids = set(p.get("detection_ids") or [])
            dets = [d for d in manual_curves(src) if d["id"] in keep_ids]
            strips = json.loads((src / "spectrograms.json").read_text())
            for png in src.glob("spec_*.png"):
                shutil.copyfile(png, job_dir / png.name)
            say(f"using {len(dets)} reviewed curves of job {src.name} (detection not re-run)")
        else:
            with ProcessPoolExecutor(min(workers, len(chunks))) as ex:
                for k, (out, strip) in enumerate(ex.map(_chunk, [(str(raw), g0, str(job_dir)) for g0 in chunks])):
                    dets += out
                    strips += [strip] if strip else []
                    say(f"finding curves: {round(100 * (k + 1) / len(chunks))}% ({k + 1} of {len(chunks)} chunks)")
        if p["search"] == "forward" and not p["detect_only"]:  # the place search scores predicted curves on this map
            say("tone map of the recording for the place search")
            M, valid = evidence.tone_map(raw, dur, workers)
            np.save(job_dir / "valid.npy", valid)
            np.save(job_dir / "map.npy", M)
            del M
    finally:
        raw.unlink(missing_ok=True)  # ~0.23 GB per hour of audio
    (job_dir / "spectrograms.json").write_text(json.dumps(strips))
    for k, d in enumerate(dets):
        d["id"] = d["id"] if p.get("detections_from") else k  # reviewed curves keep their numbers
        f = d["fit"]
        d["car_like"] = locate.car_like(f, p["car_speed_mps"], p["car_distance_m"])
    (job_dir / "detections.json").write_text(json.dumps(dets, default=float))
    trusted = [d for d in dets if d["fit"] and d["fit"]["geometry_supported"] and not d["car_like"]]
    planes = {d["group"] for d in dets if not d["car_like"]}
    (job_dir / "summary.json").write_text(json.dumps({
        "duration_s": dur, "detections": len(dets), "groups": len({d["group"] for d in dets}), "estimated_planes": len(planes),
        "trusted": len(trusted), "car_like": sum(d["car_like"] for d in dets),
        "note": "estimated planes = duplicate groups of detections that are not car-like (a provisional label)"}, indent=1))
    say(f"{len(dets)} detections (score >= {p['min_score']}), {len({d['group'] for d in dets})} duplicate groups, "
        f"{len(trusted)} trusted measurement(s), {sum(d['car_like'] for d in dets)} car-like")
    times["detect"] = time.time() - T
    if p["detect_only"]:  # plane traces only: no flight data, no search
        res = {"detect_only": True, "claim": None, "top": [], "best_score": None, "fake_sky_best": None, "n_detections": len(dets),
               "n_trusted": len(trusted), "n_groups": len({d["group"] for d in dets}), "runtime_s": times, "commit": commit}
        (job_dir / "result.json").write_text(json.dumps(res, indent=1, default=float))
        say(f"done: detection only ({len(dets)} detections, {len(trusted)} trusted measurements; see the Recording tab)")
        plog.close()
        return res

    # 3. flights
    T = time.time()
    km_lat = 111.32
    km_lon = 111.32 * np.cos(np.radians((box[0] + box[1]) / 2))
    m = float(p["margin_km"])
    keep = (box[0] - m / km_lat, box[1] + m / km_lat, box[2] - m / km_lon, box[3] + m / km_lon)
    cw = float(p["clock_s"]) if p["search"] == "tracks" else 0.0  # the curve-shape search uses the whole clock window
    t0, t1 = start - 900 - cw, start + dur + 900 + cw
    w0, w1 = t0 - 900, t1 + 900  # flights loaded with room for the curve reach
    if p.get("date_only"):
        w0, w1 = max(w0, day0), min(w1, day0 + 86399)  # that day only (the neighbouring days needn't be stored)
    days = store.days(w0, w1)
    raw, streamed = [], set()

    def sky(off):
        """The flights of [w0, w1] + off moved back by off (off = 0: the real sky; decoys and other days
        otherwise), each sky loaded for its own window (review 043); None if that window isn't covered."""
        a, b = w0 + off, w1 + off
        need = store.days(a, b)
        if p["data_mode"] == "disk":
            for day in need:
                if not store.is_stored(day):
                    say(f"{day}: storing the whole world day on disk (one stream, ~3-4 GB; reused by every later job)")
                    store.store_day(day, progress=say)
        if all(store.is_stored(day) for day in need):
            trs = store.load(keep, a, b)
        elif set(need) <= streamed:
            trs = raw
        else:
            return None
        return [type(t)(**{**t.__dict__, "t": t.t - off}) for t in flights(trs, a, b, keep)] if off else flights(trs, a, b, keep)

    if p["data_mode"] == "disk" or all(store.is_stored(day) for day in days):
        say(f"{', '.join(days)}: reading the stored world day(s) from disk (only the box and the recording's hours)")
    else:  # RAM: each day streamed, only the box kept
        # shortcut: a flight entering the box only after midnight loses its pre-midnight piece here (the
        # store keeps it); that piece is outside box + margin, so it can't reach the 9 km prediction radius
        for day in days:
            kept = []
            stream_day(day, {"box": keep}, lambda k, d: kept.append(d), progress=say)
            raw += [tr for tr in map(parse_readsb_trace, kept) if len(tr.t)]
            streamed.add(day)
            del kept
    # the date only with the curve-shape search: the whole day isn't held at once (gigabytes over a large box); each
    # search stage reads the hour it needs from the day store
    loader = None
    if p["search"] == "tracks" and p.get("date_only") and (p["data_mode"] == "disk" or all(store.is_stored(d) for d in days)):
        for day in days:
            if not store.is_stored(day):
                say(f"{day}: storing the whole world day on disk (one stream, ~3-4 GB; reused by every later job)")
                store.store_day(day, progress=say)
        loader = functools.partial(_load_flights, keep=keep, lo=w0, hi=w1)
        tracks = []
        say("flights: read from disk an hour at a time while searching (the whole day)")
    else:
        tracks = sky(0)
        say(f"{len(tracks)} aircraft active in the box (+{m:g} km margin) during the recording")
    (job_dir / "flights.json").write_text(json.dumps([{"icao": tr.icao, "type": tr.type, "reg": tr.reg, "t0": float(tr.t[0]), "t1": float(tr.t[-1]),
                                                        "n_points": len(tr.t)} for tr in tracks]))
    times["flights"] = time.time() - T

    # 4-5. candidates, scores, fake-sky controls
    T = time.time()
    elev, geoid, geoid_src = _ground(box, p["ground_elev_m"])
    hg, hm = elev + geoid, elev + geoid + float(p["mic_height_m"])
    say(f"ground {elev:.0f} m above sea level, geoid {geoid:+.1f} m ({geoid_src}): recorder at {hm:.1f} m above the WGS84 ellipsoid, for every candidate")
    lat0, lon0 = (box[0] + box[1]) / 2, (box[2] + box[3]) / 2
    groups = {}
    for d in dets:
        if p["car_filter"] and d["car_like"]:
            continue  # road traffic by our own rule: never explained by an aircraft (review 044)
        fr = np.concatenate([np.rint(np.array(q["t_s"]) / sg.HOP_S).astype(int) for q in d["contours"]])
        y = np.concatenate([1200 * np.log2(q["f_hz"]) for q in d["contours"]])
        r = np.concatenate([np.full(len(q["t_s"]), i) for i, q in enumerate(d["contours"])])
        groups.setdefault(d["group"], []).append((fr, y, r))
    det_sets = list(groups.values())
    fits = [{"id": d["id"], "t_cpa_unix": start + d["t_cpa_s"], "speed_p05": d["fit"]["speed_mps"]["p05"] - p["wind_mps"],
             "speed_p95": d["fit"]["speed_mps"]["p95"] + p["wind_mps"], "distance": d["fit"]["cpa_distance_m"]["median"],
             "distance_p05": d["fit"]["cpa_distance_m"]["p05"], "distance_p95": d["fit"]["cpa_distance_m"]["p95"], "group": d["group"]} for d in trusted]
    n = sg.n_frames(n_samples)

    def candidates(trs):
        props = locate.propose_fast(fits, trs, lat0, lon0, hg, hm, start, c)
        out = []
        for q in props:  # exact WGS84 positions from propose_fast
            if box[0] <= q["lat"] <= box[1] and box[2] <= q["lon"] <= box[3]:
                out.append({"lat": q["lat"], "lon": q["lon"], "source": "fast", "fit": q["fit"], "icao": q["icao"]})
        joint = locate.propose_joint(fits, trs, lat0, lon0, hg, hm, c, float(p["clock_s"])) if p["joint"] else []
        for q in joint:  # two trusted fits on two aircraft, one shared clock offset
            if box[0] <= q["lat"] <= box[1] and box[2] <= q["lon"] <= box[3]:
                out.append({"lat": q["lat"], "lon": q["lon"], "source": "joint", "fit": q["fit"], "icao": q["icao"],
                            "proposal_clock_s": q["clock_s"], "gap_m": q["gap_m"]})
        if p["grid_m"] and float(p["grid_m"]) > 0:
            # grid in exact local metres around the box centre, kept where it falls inside the box
            g = float(p["grid_m"])
            (w, s_), (e_, n_) = locate.to_local(box[0], box[2], lat0, lon0, hm), locate.to_local(box[1], box[3], lat0, lon0, hm)
            for ee in np.arange(min(w, e_), max(w, e_) + g, g):
                for nn in np.arange(min(s_, n_), max(s_, n_) + g, g):
                    la, lo = locate.to_geo(ee, nn, lat0, lon0, hm)
                    if box[0] <= la <= box[1] and box[2] <= lo <= box[3]:
                        out.append({"lat": la, "lon": lo, "source": "grid"})
        return out, (len(props), len(joint))

    # spatial index over the widest window any sky uses (fake skies shift times, not positions)
    make_index = lambda trs: locate.TileIndex(trs, lat0, lon0, hm, w0, w1) if p.get("index", True) else None
    index = make_index(tracks) if p["search"] != "tracks" else None  # the curve-shape search needs no index (and a whole day of a
    # large box takes many minutes to index)
    miss = float(p["silence_penalty"])
    rank = lambda q: (bool(q.get("matches")), q.get("score", 0.0))  # places explaining no curve rank last

    def score_all(trs, label, index=index):
        cands, n_props = candidates(trs)
        say(f"{label}: {n_props[0]} fast-path and {n_props[1]} joint proposals, {len(cands)} candidate places in the box; scoring on {workers} worker(s)")
        with ProcessPoolExecutor(workers, initializer=_init, initargs=(det_sets, n, start, c, hg, hm, float(p["clock_s"]), miss, bool(p["flying_only"]))) as ex:
            def score(qs):
                k = max(1, len(qs) // (8 * workers))  # batch size: keeps every worker busy
                tiles = {}
                for q in qs:  # grouped by tile: each batch sees exactly its places' near() tracks
                    tiles.setdefault(None if index is None else index.key(q["lat"], q["lon"]), []).append(q)
                batches = [b[i:i + k] for b in tiles.values() for i in range(0, len(b), k)]
                tasks = [(b, trs if index is None else [trs[i] for i in index.near(b[0]["lat"], b[0]["lon"])]) for b in batches]
                for b, res in zip(batches, ex.map(_score, tasks)):
                    for q, (s, clock, pairs, unheard) in zip(b, res):
                        q.update(score=s, clock_s=clock, matches=[{"group": i, "icao": ic, "residual_cents": r} for i, ic, r in pairs], unheard=unheard)
            score(cands)
            if p["refine"] and cands:  # the same budget and rules in every sky (review 042)
                for step, half in ((250.0, 1000.0), (100.0, 200.0)):  # coarse, then fine around the new best
                    pts = refine_points(sorted(cands, key=rank, reverse=True), step, half)
                    say(f"{label}: refining, {len(pts)} more places at {step:g} m")
                    score(pts)
                    cands += pts
        return sorted(cands, key=rank, reverse=True)

    def refine_points(ranked, step, half):
        """Grid points (step m, within +-half m, inside the box) around the REFINE_N best distinct places
        that explain a curve (at least 1 km apart)."""
        tops = []
        for q in ranked:
            if q.get("matches") and all(_dist(q, t) >= 1000.0 for t in tops):
                tops.append(q)
            if len(tops) == REFINE_N:
                break
        out = []
        for t in tops:
            ce, cn = locate.to_local(t["lat"], t["lon"], lat0, lon0, hm)
            for a in np.arange(-half, half + 1e-6, step):
                for b in np.arange(-half, half + 1e-6, step):
                    if a == 0 and b == 0:
                        continue
                    la, lo = locate.to_geo(ce + a, cn + b, lat0, lon0, hm)
                    if box[0] <= la <= box[1] and box[2] <= lo <= box[3]:
                        out.append({"lat": la, "lon": lo, "source": "refine", "around": [t["lat"], t["lon"]]})
        return out

    def forward():
        """The place search (doppler/search.py) in the real sky and one fake sky: every curve x every propeller flight
        proposes places; the best are checked by scoring each nearby flight's predicted curve on the tone map."""
        ridges = []
        for d in dets:
            cs = [q for q in d["contours"] if len(q["t_s"]) >= 8]
            if cs and not (p["car_filter"] and d["car_like"]):
                ridges.append((d["id"], *search.ridge_points([{"t": start + np.asarray(q["t_s"]), "f": q["f_hz"]} for q in cs])))
        frame = (lat0, lon0, hm)
        prog = lambda label: (lambda stage, k, tot: say(f"{label}: {stage} {k}/{tot}"))
        as_cand = lambda q: {"lat": q["lat"], "lon": q["lon"], "score": q["score"], "clock_s": q["clock_s"], "source": "search", "T": q["T"],
                             "n_passes": q["n_passes"], "two_flight": q["two_flight"],
                             "matches": [{"icao": h[0], "type": h[1], "km": h[2], "evidence": h[3]} for h in q["heard"]]}
        say(f"real sky: place search, {len(ridges)} curves x the propeller flights, clock +-{float(p['clock_s']):g} s")
        found, _, info = search.search_places(job_dir, ridges, tracks, start, frame, hg, c, box, float(p["clock_s"]), workers, say=prog("real sky"))
        say(f"real sky: {info['n_cells']} proposed cells ({info['n_two_flight']} backed by two flights), {info['n_verified']} places checked")
        fake = {}
        if not p["fake_sky"]:
            say("fake-sky test skipped (unticked): no comparison with chance")
        else:  # one fake sky, far enough that no recorded plane is in it: the first whose flights are loaded
            shift = float(p["null_shift_s"]) or max(900.0, 2 * float(p["clock_s"]) + dur + 600.0)
            for off in (-shift, shift):
                trs = sky(off)
                if trs is not None:
                    say(f"fake sky (the flights of {off / 60:+.0f} min): the same place search")
                    fake[f"{off:+.0f}s"] = [as_cand(q) for q in search.search_places(job_dir, ridges, trs, start, frame, hg, c, box, float(p["clock_s"]),
                                                                                        workers, say=prog("fake sky"))[0]]
                    break
            else:
                say("fake sky skipped: no other time's flights are loaded (disk mode stores them)")
        return [as_cand(q) for q in found], fake

    chance = {}  # the curve-shape search's free chance level, date only

    def by_tracks():
        """The track search (doppler/trackmatch.py) in the real sky and one fake sky: every curve's points against
        every flight's real track, at every place of the box (turns, descents and jets included)."""
        use = [d for d in dets if not (p["car_filter"] and d["car_like"])]
        frame = (lat0, lon0, hm)
        run_ = lambda trs, label, ld=None: polish(trackmatch.search(
            use, trs, start, box, frame, hg, c, float(p["clock_s"]), workers, float(p["wind_mps"]), p["top_n"],
            say=lambda m: say(f"{label}: {m}"), loader=ld, work_dir=job_dir / "slices" / label.replace(" ", "_"),
            info=chance if label == "real sky" else None), trs, label, ld)

        def polish(found, trs, label, ld=None):
            """The best place searched again POLISH_KM around it with the ground height there (the search took the box
            centre's: off by 100+ m in a large box), the clock within POLISH_CLOCK_S of the one it found; the places
            farther away keep their rank below. Development: the video's camera 115 -> 115 m, an Eclipse recorder 275 ->
            117 m in a box over most of the US (scripts/polish_test.py)."""
            if not found:
                return found
            b0 = found[0]
            dlat = POLISH_KM / 111.32
            dlon = dlat / np.cos(np.radians(b0["lat"]))
            small = (b0["lat"] - dlat, b0["lat"] + dlat, b0["lon"] - dlon, b0["lon"] + dlon)
            elev_p, geoid_p, _ = _ground(small, p["ground_elev_m"])  # "auto": looked up there; a number: kept
            hg_p = elev_p + geoid_p
            say(f"{label}: polishing around the best place, ground {elev_p:.0f} m above sea level there")
            if ld is not None:  # the flights around the clock found, read for this step only
                trs = ld(start + b0["clock_s"] - 900.0, start + b0["clock_s"] + dur + 900.0)
            near = trackmatch.search(use, trs, start + b0["clock_s"], small, (b0["lat"], b0["lon"], hg_p + float(p["mic_height_m"])),
                                     hg_p, c, POLISH_CLOCK_S, workers, float(p["wind_mps"]), p["top_n"], say=lambda m: say(f"{label}: polish: {m}"))
            for q in near:
                q["clock_s"] += b0["clock_s"]
            far = [q for q in found if abs(q["lat"] - b0["lat"]) > dlat or abs(q["lon"] - b0["lon"]) > dlon]
            return (near + far)[:p["top_n"]]

        say(f"real sky: track search, {len(use)} curves x every flight's real track, clock +-{float(p['clock_s']):g} s")
        found, fake = run_(tracks, "real sky", loader), {}
        if chance.get("chance_score") is not None:
            say(f"chance level (date only, free): the best place at another start time or 30+ km away scored "
                f"{chance['chance_score']:.1f}, the winner {chance['winner_same_stage'] or 0:.1f} at that same stage")
        if not p["fake_sky"]:
            say("fake-sky test skipped (unticked): no comparison with chance")
        else:
            shift = float(p["null_shift_s"]) or max(900.0, 2 * float(p["clock_s"]) + dur + 600.0)
            for off in (-shift, shift):
                if loader is not None:  # read from disk as the real sky is, if that time's days are stored
                    ok = all(store.is_stored(d) for d in store.days(w0 + off, w1 + off))
                    trs, ld = ([], functools.partial(_load_flights, keep=keep, lo=w0, hi=w1, off=off)) if ok else (None, None)
                else:
                    trs, ld = sky(off), None
                if trs is not None:
                    say(f"fake sky (the flights of {off / 60:+.0f} min): the same track search")
                    fake[f"{off:+.0f}s"] = run_(trs, "fake sky", ld)
                    break
            else:
                say("fake sky skipped: no other time's flights are loaded (disk mode stores them)")
        return found, fake

    if p["search"] == "forward":
        real, nulls = forward()
    elif p["search"] == "tracks":
        real, nulls = by_tracks()
    else:
        real = score_all(tracks, "real sky")
        nulls = {}
        if not p["fake_sky"]:
            say("fake-sky test skipped (unticked): no comparison with chance")
        else:
            # decoys as dopplerprint's: far enough that no recorded plane can be in them (review 043)
            shift = float(p["null_shift_s"]) or max(900.0, 2 * float(p["clock_s"]) + dur + 600.0)
            offs = [(f"{dt:+.0f}s", dt, f"fake sky (the flights of {dt / 60:+.0f} min)") for dt in (shift, -shift)]
            days_off = p["other_day_null"] if isinstance(p["other_day_null"], list) else [p["other_day_null"]]
            for d in (int(v) for v in days_off if int(v)):  # one or several other days' skies
                offs.append((f"{d:+d}d", d * 86400.0, f"other-day sky ({', '.join(store.days(w0 + d * 86400, w1 + d * 86400))}, same time of day)"))
            for key, off, label in offs:
                trs = sky(off)
                if trs is None:
                    say(f"{label} skipped: its flight data isn't loaded (RAM mode streams only the recording's day(s); use disk mode)")
                    continue
                nulls[key] = score_all(trs, label, make_index(trs))
    (job_dir / "candidates.json").write_text(json.dumps(real, default=float))
    (job_dir / "nulls.json").write_text(json.dumps({k: v[:p["top_n"]] for k, v in nulls.items()}, default=float))
    times["search"] = time.time() - T

    # 6. result
    null_best = max((v[0]["score"] for v in nulls.values() if v and v[0].get("matches")), default=0.0) if p["fake_sky"] else None
    top = real[0] if real and real[0].get("matches") else None
    # review 040: two fake skies can't give a false-claim rate, so no claim until calibrated
    beats = bool(top and top["score"] > null_best + p["claim_margin"]) if p["fake_sky"] else None
    types = {tr.icao: tr.type for tr in tracks}
    locate_idx = lambda q: tracks if index is None else [tracks[i] for i in index.near(q["lat"], q["lon"])]
    for q in real[:p["top_n"]]:
        for mm in q.get("matches", []):
            mm["type"] = types.get(mm["icao"], mm.get("type"))  # the curve-shape search names its own
        if p["flying_only"] and p["search"] != "tracks":  # kept for audit (review 044): targets the flying rule ruled out at this place
            preds = locate.predictions(locate_idx(q), q["lat"], q["lon"], hm, start, n, c, hg)
            q["excluded_slow"] = [{"icao": icao, "speed_mps": info["speed_mps"]} for icao, _, _, info in preds if info["speed_mps"] < locate.MIN_FLYING_MPS]
    res = {"claim": None, "claim_note": "not calibrated: the fake-sky comparison is information only", "beats_fake_sky": beats,
           "best_score": top and top["score"], "fake_sky_best": null_best, "margin": p["claim_margin"],
           "top": real[:p["top_n"]], "n_candidates": len(real), "n_detections": len(dets), "n_trusted": len(trusted),
           "n_aircraft": len(tracks), "days": days, "runtime_s": times, "commit": commit,
           "coordinates": {"positions": "WGS84 lat/lon (degrees)", "heights": "metres above the WGS84 ellipsoid",
                           "ground_elev_m": elev, "geoid_m": geoid, "geoid_source": geoid_src, "time": "UTC"},
           "search": p["search"], "chance_elsewhere": chance or None,
           "note": None if trusted or p["grid_m"] or p["search"] in ("forward", "tracks") else "no trusted measurement and no grid: nothing to propose"}
    (job_dir / "result.json").write_text(json.dumps(res, indent=1, default=float))
    if p["fake_sky"]:
        say(f"done: best score {top['score'] if top else 0:.2f} vs fake sky {null_best:.2f} ({'beats' if beats else 'does not beat'} it by "
            f"{p['claim_margin']:g}; not a calibrated claim)")
    else:
        say(f"done: best score {top['score'] if top else 0:.2f} (no fake-sky test)")
    plog.close()
    return res
