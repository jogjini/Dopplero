"""The desktop app's local analysis service: upload a recording, set the time and a search box, and run the
localisation pipeline (doppler.locate_job). Standard library only; serves on this machine only (127.0.0.1:8765).
The desktop app starts it by itself; to run it alone: uv run python tools/locate_app.py

Jobs run one at a time, in the background; each keeps everything it measured and found in
outputs/locate_jobs/<id>/ (params, audio, progress log, detections, flights, candidates, fake-sky
controls, result). A finished job can be saved as one .zip (results and logs, not the audio) and
loaded back later. Use only on recordings you have the right to locate.
"""
import io
import json
import math
import queue
import re
import sys
import threading
import time
import traceback
import uuid
import zipfile
from datetime import datetime
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from doppler import locate_job, memguard  # noqa: E402
from doppler.adsb import store  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "outputs" / "locate_jobs"
HOST, PORT = "127.0.0.1", 8765
MAX_UPLOAD = 2 * 1024 ** 3
FILES = {"summary.json", "detections.json", "spectrograms.json", "candidates.json", "flights.json", "result.json", "params.json", "curation.json", "manual.json"}
SAVED = FILES | {"audio.json", "nulls.json", "progress.log", "request.json", "error.txt"}  # what a saved result holds (not the audio)
MAX_UNZIPPED = 1024 ** 3
STATE = {}          # job id -> {"status", "log": [...], "error"}
TODO = queue.Queue()
RENDER = threading.BoundedSemaphore(2)
WARMING = set()  # recordings whose viewer decode was started (locate_job.raw_audio is a no-op once cached)


def _run_job(jid):
    """One job, in a child process that leads its own process group: Stop ends it and every worker process it
    started (a pool's workers would otherwise be left running as orphans). Progress goes to the job's progress.log."""
    import os
    os.setsid()
    params = json.loads((JOBS / jid / "request.json").read_text())
    try:
        locate_job.run(JOBS / jid, params, log=lambda line: None)
    except Exception:
        (JOBS / jid / "error.txt").write_text(traceback.format_exc())
        raise SystemExit(1)


def worker():
    import multiprocessing
    while True:
        jid = TODO.get()
        st = STATE[jid]
        if st["status"] == "stopped":  # stopped while it waited in the queue
            continue
        st["status"] = "running"
        proc = multiprocessing.get_context("spawn").Process(target=_run_job, args=(jid,))
        proc.start()
        st["proc"] = proc
        while proc.is_alive():  # last resort before the Mac freezes (the search itself gives back workers first)
            proc.join(timeout=1.0)
            free = memguard.free_pct()
            if proc.is_alive() and free is not None and free < memguard.HARD_PCT:
                stop_job(jid, f"stopped: memory nearly full ({free} % free); try fewer CPU workers or a smaller box")
        if st["status"] == "stopped":
            continue
        if proc.exitcode == 0 and (JOBS / jid / "result.json").exists():
            st["status"] = "done"
        else:
            err = JOBS / jid / "error.txt"
            st["status"], st["error"] = "failed", err.read_text().strip().splitlines()[-1] if err.exists() else f"the job's process ended with code {proc.exitcode}"


def check_params(params):
    """A job's parameters as sent by the page, checked (ValueError / AssertionError / KeyError otherwise)."""
    params["detect_only"] = bool(params.get("detect_only"))
    params["detector"] = "manual"  # the curves are marked by hand: no automatic detection
    assert params["detect_only"] or params.get("detections_from"), \
        "open the recording first (curves only), mark its curves, then search with them"
    params["search"] = params.get("search") or "forward"
    assert params["search"] in ("forward", "fits", "tracks"), "search: forward (new), fits (old) or tracks (curve shapes vs real tracks)"
    if params.get("start_utc") or not params["detect_only"]:
        datetime.fromisoformat(params["start_utc"].replace("Z", "+00:00"))
    if not params["detect_only"]:  # traces only: no box needed
        box = [float(v) for v in params["box"]]
        assert len(box) == 4 and box[0] < box[1] and box[2] < box[3] and -90 <= box[0] and box[1] <= 90, "draw a search box"
    return params


MAX_MANUAL_POINTS = 2_000_000


def save_manual(jid, curves):
    """The curves the user fitted by hand (the desktop app's curve tool), stored as sent once checked: each a model
    name, its parameters and points (to edit it again), and its lines as contours (what locating uses)."""
    assert isinstance(curves, list) and len(curves) <= 1000, "curves: a list of at most 1000"
    finite = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    total = 0
    for cv in curves:
        assert isinstance(cv, dict) and isinstance(cv.get("model"), str) and len(cv["model"]) <= 40, "each curve needs a model"
        assert isinstance(cv.get("params"), dict) and all(finite(v) for v in cv["params"].values()), "parameters: numbers"
        ext = cv.get("extent")
        assert isinstance(ext, list) and len(ext) == 2 and all(map(finite, ext)) and ext[0] < ext[1], "extent: [start, end]"
        assert isinstance(cv.get("contours"), list), "contours: a list"
        for q in cv["contours"]:
            t, f = q.get("t_s"), q.get("f_hz")
            assert isinstance(t, list) and isinstance(f, list) and len(t) == len(f), "contour: t_s and f_hz of one length"
            assert all(map(finite, t)) and all(finite(v) and 0 < v <= 24000 for v in f), "contour: finite times, frequencies"
            total += len(t)
        ps = cv.get("pass")
        assert ps is None or (isinstance(ps, dict) and all(finite(ps.get(k)) for k in ("t_cpa_s", "f0_hz", "speed_mps", "distance_m"))), \
            "pass: t_cpa_s, f0_hz, speed_mps, distance_m"
        assert cv.get("residual_cents") is None or finite(cv["residual_cents"]), "residual_cents: a number"
        total += len(cv.get("points") or [])
    assert total <= MAX_MANUAL_POINTS, "too many points"
    (JOBS / jid / "manual.json").write_text(json.dumps(curves))
    return len(curves)


def relocate(jid, settings=None):
    """A new job on the same recording, locating with the curves the user marked by hand (manual.json). A locating
    job keeps its own box and settings; a curves-only job takes them from the app (settings: its search form)."""
    import os
    import shutil
    d = JOBS / jid
    if STATE.get(jid, {}).get("status") in ("queued", "running"):
        raise ValueError("the job is still running")
    req = json.loads((d / "request.json").read_text())
    if settings:  # the page's form (time, box, search ...) over the job's own, for any job
        req = {**req, **settings, "detect_only": False}
    elif req.get("detect_only") or not req.get("box"):
        raise ValueError("this job only found traces: set the box, time and settings in the form, then locate")
    req = check_params({**req, "detections_from": str(d)})
    audio = next(d.glob("input.*"), None)
    if audio is None or not (d / "detections.json").exists():
        raise ValueError("this job's audio or curves aren't kept")
    dets = [x for x in locate_job.manual_curves(d) if x.get("manual")]  # the curves marked by hand
    ids = [x["id"] for x in dets]
    if not ids:
        raise ValueError("mark at least one curve first")
    nid = new_id()
    nd = JOBS / nid
    nd.mkdir(parents=True)
    os.symlink(audio.resolve(), nd / audio.name)  # the same recording, not a copy
    req.update(audio_path=str(nd / audio.name), detections_from=str(d), detection_ids=ids,
               file_name=f"{json.loads((d / 'request.json').read_text()).get('file_name') or audio.name} ({len(ids)} curves)")
    (nd / "request.json").write_text(json.dumps(req, indent=1))
    # the curves and the spectrogram strips in the new folder now, so the app shows them while the job waits or runs
    # (the job writes the same files again when it starts)
    (nd / "detections.json").write_text(json.dumps(dets))
    for name in ("spectrograms.json", "summary.json"):
        if (d / name).exists():
            shutil.copyfile(d / name, nd / name)
    if (nd / "summary.json").exists():
        sm = json.loads((nd / "summary.json").read_text())
        sm.update(detections=len(dets), estimated_planes=len({x["group"] for x in dets if not x.get("car_like")}),
                  trusted=sum(bool(x.get("fit") and x["fit"].get("geometry_supported") and not x.get("car_like")) for x in dets),
                  car_like=sum(bool(x.get("car_like")) for x in dets))
        (nd / "summary.json").write_text(json.dumps(sm, indent=1))
    for png in d.glob("spec_*.png"):
        shutil.copyfile(png, nd / png.name)
    STATE[nid] = {"status": "queued", "log": [f"queued ({TODO.qsize()} job(s) ahead)"]}
    TODO.put(nid)
    return nid


def stop_job(jid, why="stopped by the user"):
    """Stop a queued or running job: a queued one is skipped, a running one is ended with all its worker processes."""
    import os
    import signal
    st = STATE.get(jid)
    if not st or st["status"] not in ("queued", "running"):
        return False
    was_running = st["status"] == "running"
    st["status"] = "stopped"
    proc = st.get("proc")
    if was_running and proc is not None and proc.pid:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                break
            except PermissionError:  # macOS refuses the group call if any member can't take the signal; the rest got it
                pass
            proc.join(timeout=5)
            if not proc.is_alive():
                break
    (JOBS / jid / "audio.f32").unlink(missing_ok=True)  # the decoded audio of an unfinished job (large)
    (JOBS / jid / "stopped.txt").write_text(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + f" {why}\n")
    with open(JOBS / jid / "progress.log", "a") as fh:
        fh.write(f"{datetime.now():%H:%M:%S} {why}\n")
    return True


def new_id():
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


def saved_name(name):
    return name in SAVED or re.fullmatch(r"spec_\d{7}(_p)?\.png", name)


def export_job(jid):
    """One job's results and logs as a zip (the uploaded audio is left out)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted((JOBS / jid).iterdir()):
            if f.is_file() and saved_name(f.name):
                z.write(f, f.name)
    return buf.getvalue()


def import_job(data):
    """A zip from export_job back in as a new job; returns its id. Only the known flat file names
    are accepted (no paths), within MAX_UNZIPPED in total."""
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise ValueError("not a zip file")
    members = z.infolist()
    bad = [m.filename for m in members if not saved_name(m.filename)]
    if bad:
        raise ValueError(f"not a saved result (unexpected files: {', '.join(bad[:3])})")
    if not any(m.filename in ("result.json", "error.txt") for m in members):
        raise ValueError("not a saved result (no result.json)")
    if sum(m.file_size for m in members) > MAX_UNZIPPED:
        raise ValueError("too large")
    jid = new_id()
    d = JOBS / jid
    d.mkdir(parents=True)
    for m in members:
        (d / m.filename).write_bytes(z.read(m))
    return jid


def flight_path(jid, icao):
    """One aircraft's reported positions while the job's recording ran (+-5 min), read back from the
    stored world day with the job's box and margin; LookupError if that day isn't stored."""
    p = json.loads((JOBS / jid / "params.json").read_text())
    dur = json.loads((JOBS / jid / "audio.json").read_text())["duration_s"]
    start = datetime.fromisoformat(p["start_utc"].replace("Z", "+00:00")).timestamp()
    w0, w1 = start - 300, start + dur + 300
    if not all(store.is_stored(day) for day in store.days(w0, w1)):
        raise LookupError("this day's flight data isn't stored on disk (the job streamed it)")
    b, m = p["box"], float(p["margin_km"])
    dlon = m / (111.32 * max(math.cos(math.radians((b[0] + b[1]) / 2)), 0.01))
    keep = (b[0] - m / 111.32, b[1] + m / 111.32, b[2] - dlon, b[3] + dlon)  # as the job's
    for tr in locate_job.flights(store.load(keep, w0, w1), w0, w1, keep):
        if tr.icao == icao:
            return {"icao": icao, "type": tr.type, "reg": tr.reg, "t_rec_s": (tr.t - start).tolist(), "lat": tr.lat.tolist(),
                    "lon": tr.lon.tolist(), "alt_m": [None if not math.isfinite(a) else a for a in tr.alt_m.tolist()]}
    raise LookupError("no such aircraft in this job's flight data")


def rerun_job(jid):
    """Queue a stopped or failed job again in its own folder: the same request (settings and curves), and the search's
    finished slices are reused (only the rest is redone)."""
    d = JOBS / jid
    st = STATE.get(jid)
    if (st and st["status"] in ("queued", "running")) or not (d / "request.json").exists() or (d / "result.json").exists():
        return False
    for name in ("stopped.txt", "error.txt"):
        (d / name).unlink(missing_ok=True)
    with open(d / "progress.log", "a") as fh:
        fh.write(f"{datetime.now():%H:%M:%S} run again with the same settings and curves\n")
    STATE[jid] = {"status": "queued", "log": [f"queued ({TODO.qsize()} job(s) ahead)"]}
    TODO.put(jid)
    return True


def job_view(jid):
    d = JOBS / jid
    st = STATE.get(jid) or {"status": "stopped" if (d / "stopped.txt").exists() else "done" if (d / "result.json").exists()
                            else "failed" if (d / "error.txt").exists() else "unknown", "log": []}
    log = (d / "progress.log").read_text().splitlines() if (d / "progress.log").exists() else st["log"]  # the job's own progress lines
    out = {"id": jid, "status": st["status"], "log": log[-200:], "error": st.get("error"), "has_audio": any(d.glob("input.*"))}
    for name in ("params", "result"):
        f = d / f"{name}.json"
        if f.exists():
            out[name] = json.loads(f.read_text())
    if out["status"] == "failed" and not out["error"] and (d / "error.txt").exists():
        out["error"] = (d / "error.txt").read_text().splitlines()[-1]
    return out


def job_list(ids):
    """The jobs for the app's list: each with its recording (root: the job whose curves it was located with, followed
    back to the one that loaded the audio), the settings it ran with, and its result's scores against chance."""
    reqs = {}

    def req(i):
        if i not in reqs:
            f = JOBS / i / "request.json"
            reqs[i] = json.loads(f.read_text()) if f.exists() else {}
        return reqs[i]

    def root(i):
        seen = {i}
        while (src := req(i).get("detections_from")) and (p := Path(src).name) not in seen and (JOBS / p).is_dir():
            seen.add(p)
            i = p
        return i

    out = []
    for i in ids:
        r = req(i)
        row = {"id": i, "status": job_view(i)["status"], "file": r.get("file_name"), "root": root(i),
               "detect_only": bool(r.get("detect_only")),
               "settings": {k: r.get(k) for k in ("start_utc", "date_only", "clock_s", "box", "search", "workers", "fake_sky", "detector", "data_mode")}
               | {"curves": len(r["detection_ids"]) if r.get("detection_ids") else None}}
        f = JOBS / i / "result.json"
        if f.exists():
            res = json.loads(f.read_text())
            row["verdict"] = {k: res.get(k) for k in ("best_score", "chance_elsewhere", "fake_sky_best", "beats_fake_sky", "detect_only")}
        out.append(row)
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, default=float).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass

    def allowed(self, write=False):
        """Only this machine's own page (aircraft-doppler's guard): the Host header must name this server
        (blocks DNS rebinding); writes must carry X-Locate-App, which another site's page can't add
        without a CORS preflight that this server never answers, and a foreign Origin is refused."""
        hosts = {f"{h}:{PORT}" for h in ("127.0.0.1", "localhost")}
        if self.headers.get("Host") not in hosts:
            return False
        origin = self.headers.get("Origin")
        if write and (self.headers.get("X-Locate-App") != "1" or (origin is not None and origin not in {f"http://{h}" for h in hosts})):
            return False
        return True

    def do_GET(self):
        if not self.allowed():
            return self._send(403, {"error": "forbidden"})
        if self.path == "/api/jobs":
            ids = sorted((p.name for p in JOBS.glob("*") if (p / "request.json").is_file()), reverse=True) if JOBS.exists() else []
            return self._send(200, job_list(ids[:500]))
        if self.path.startswith("/api/jobs/") and "/track/" in self.path:
            parts = self.path.split("/")  # /api/jobs/<id>/track/<icao>
            jid, icao = parts[3], parts[5] if len(parts) > 5 else ""
            if not re.fullmatch(r"[0-9a-f-]+", jid) or not re.fullmatch(r"~?[0-9a-f]{6}", icao) or not (JOBS / jid / "params.json").is_file():
                return self._send(404, {"error": "no such job or aircraft"})
            try:
                return self._send(200, flight_path(jid, icao))
            except LookupError as e:
                return self._send(404, {"error": str(e)})
        if self.path.startswith("/api/jobs/") and self.path.endswith("/export"):
            jid = self.path.split("/")[3]
            if not re.fullmatch(r"[0-9a-f-]+", jid) or not (JOBS / jid).is_dir():
                return self._send(404, {"error": "no such job"})
            if STATE.get(jid, {}).get("status") in ("queued", "running"):
                return self._send(409, {"error": "the job is still running"})
            data = export_job(jid)
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", f'attachment; filename="locate_{jid}.zip"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            return self.wfile.write(data)
        if self.path.startswith("/api/jobs/") and "/render?" in self.path:
            jid = self.path.split("/")[3]
            q = parse_qs(urlparse(self.path).query)
            try:
                v = {k: float(q[k][0]) for k in ("t0", "t1", "fmin", "fmax", "win", "pps")}
                axis = q.get("axis", ["lin"])[0]
                height = int(float(q.get("h", ["330"])[0]))
                assert re.fullmatch(r"[0-9a-f-]+", jid) and axis in ("lin", "log") and v["win"] in locate_job.sg.WINDOWS_S
                # the continuous viewer asks for its exact window: any span up to 1 h, 0.02-8 columns per second (the
                # browser stretches beyond 8: the analysis has 8 frames per second), its own height
                assert 0 <= v["t0"] < v["t1"] <= v["t0"] + 3600 and 15 <= v["fmin"] < v["fmax"] <= 7500 and 0.02 <= v["pps"] <= 8
                assert 150 <= height <= 1200
            except (KeyError, ValueError, AssertionError):
                return self._send(400, {"error": "bad render parameters"})
            audio = next((JOBS / jid).glob("input.*"), None) if (JOBS / jid).is_dir() else None
            if audio is None:
                return self._send(404, {"error": "no audio kept for this job (a loaded result): fixed images only"})
            with RENDER:  # a few at a time: each decodes and analyses up to an hour of audio
                png = locate_job.render(audio, v["t0"], v["t1"], v["fmin"], v["fmax"], axis, v["win"], v["pps"], height)
            return self._send(200, png, "image/png")
        if self.path.startswith("/api/jobs/") and "/files/" in self.path:
            parts = self.path.split("/")  # /api/jobs/<id>/files/<name>
            jid, name = parts[3], parts[5] if len(parts) > 5 else ""
            ok = re.fullmatch(r"[0-9a-f-]+", jid) and (name in FILES or re.fullmatch(r"spec_\d{7}(_p)?\.png", name))
            f = JOBS / jid / name
            if not ok or not f.is_file():
                return self._send(404, {"error": "no such file"})
            return self._send(200, f.read_bytes(), "image/png" if name.endswith(".png") else "application/json")
        if self.path.startswith("/api/jobs/"):
            jid = self.path.split("/")[3]
            if not (JOBS / jid).is_dir() or "/" in jid or ".." in jid:
                return self._send(404, {"error": "no such job"})
            audio = next((JOBS / jid).glob("input.*"), None)
            if audio is not None and audio not in WARMING:  # decoded for the viewer while the user looks at the first image
                WARMING.add(audio)
                threading.Thread(target=locate_job.raw_audio, args=(audio,), daemon=True).start()
            return self._send(200, job_view(jid))
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self.allowed(write=True):
            return self._send(403, {"error": "forbidden"})
        if self.path.startswith("/api/jobs/") and any(self.path.endswith(e) for e in ("/relocate", "/manual")):
            jid = self.path.split("/")[3]
            if not re.fullmatch(r"[0-9a-f-]+", jid) or not (JOBS / jid / "detections.json").is_file():
                return self._send(404, {"error": "no such job, or it has no curves yet"})
            n = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(n)) if 0 < n <= (50_000_000 if self.path.endswith("/manual") else 1_000_000) else {}
                if self.path.endswith("/manual"):
                    return self._send(200, {"saved": save_manual(jid, body.get("curves"))})
                return self._send(200, {"id": relocate(jid, body.get("settings"))})
            except (ValueError, AssertionError, TypeError, KeyError) as e:
                return self._send(400, {"error": str(e)})
        if self.path.startswith("/api/jobs/") and self.path.endswith("/stop"):
            jid = self.path.split("/")[3]
            if not re.fullmatch(r"[0-9a-f-]+", jid) or jid not in STATE:
                return self._send(404, {"error": "no such job in this session"})
            return self._send(200, {"stopped": stop_job(jid)})
        if self.path.startswith("/api/jobs/") and self.path.endswith("/rerun"):
            jid = self.path.split("/")[3]
            if not re.fullmatch(r"[0-9a-f-]+", jid) or not (JOBS / jid).is_dir():
                return self._send(404, {"error": "no such job"})
            if not rerun_job(jid):
                return self._send(400, {"error": "only a stopped or failed job can be run again"})
            return self._send(200, {"id": jid})
        if self.path not in ("/api/jobs", "/api/import"):
            return self._send(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", 0))
        if not 0 < n <= MAX_UPLOAD:
            return self._send(413, {"error": "upload missing or too large"})
        if self.path == "/api/import":
            try:
                return self._send(200, {"id": import_job(self.rfile.read(n))})
            except ValueError as e:
                return self._send(400, {"error": str(e)})
        msg = BytesParser(policy=policy.default).parsebytes(
            b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + self.rfile.read(n))
        parts = {p.get_param("name", header="content-disposition"): p for p in msg.iter_parts()}
        if "audio" not in parts or "params" not in parts:
            return self._send(400, {"error": "need an audio file and parameters"})
        try:
            params = check_params(json.loads(parts["params"].get_content()))
        except Exception as e:
            return self._send(400, {"error": f"bad parameters: {e}"})
        jid = new_id()
        d = JOBS / jid
        d.mkdir(parents=True)
        fname = Path(parts["audio"].get_filename() or "audio.bin").name
        audio = d / ("input" + Path(fname).suffix.lower())
        audio.write_bytes(parts["audio"].get_payload(decode=True))
        params.update(audio_path=str(audio), file_name=fname)
        (d / "request.json").write_text(json.dumps(params, indent=1))
        STATE[jid] = {"status": "queued", "log": [f"queued ({TODO.qsize()} job(s) ahead)"]}
        TODO.put(jid)
        self._send(200, {"id": jid})


def main():
    import os
    os.chdir(ROOT)  # the pipeline's data paths (data/adsb, git) are relative to the project
    JOBS.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=worker, daemon=True).start()
    if os.environ.get("DOPPLER_STOP_WITH_PARENT"):  # started by the desktop app: stop when it closes, however it ends
        parent = os.getppid()

        def watch():
            while os.getppid() == parent:
                time.sleep(1)
            stop_workers()
        threading.Thread(target=watch, daemon=True).start()
    print(f"serving on http://{HOST}:{PORT}  (jobs in {JOBS})", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


def stop_workers(*_):
    """Stopping the server stops a running job's worker processes too (none left behind)."""
    import multiprocessing
    import os
    import signal
    for st in STATE.values():  # a running job leads its own process group: end it with its workers
        proc = st.get("proc")
        if proc is not None and proc.is_alive():
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    for p in multiprocessing.active_children():
        p.terminate()
    os._exit(0)


if __name__ == "__main__":
    import multiprocessing
    import signal
    multiprocessing.set_start_method("spawn", force=True)
    signal.signal(signal.SIGINT, stop_workers)
    signal.signal(signal.SIGTERM, stop_workers)
    main()
