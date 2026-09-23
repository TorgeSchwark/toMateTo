#!/usr/bin/env python3
"""Local web GUI for nnue-chess training: start/track full_cycle and
self_play runs (including several at once, each with its own --threads
budget), without touching a terminal.

Deliberately stdlib-only (http.server + subprocess + threading) - no pip
install required, so this runs anywhere the training binaries themselves
already run (WSL/Linux).

Usage:
    python3 gui/server.py [--port 8787]
    -> open http://localhost:8787 in a browser (WSL2 forwards localhost to
       Windows automatically, so this also works from a normal Windows
       browser without extra setup).

Does NOT rebuild anything by default. Architecture (ACC_SIZE/H1/H2/H3) is
whatever build/ is currently configured for (see /api/status); use the
"Architektur neu bauen" panel in the UI to reconfigure + rebuild - that
action is serialized against job starts (see BUILD_LOCK below) since both
sides touch the same build/ directory and the same binaries.
"""
import csv
import json
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

NNUE_CHESS_DIR = Path(__file__).resolve().parent.parent
BUILD_DIR = NNUE_CHESS_DIR / "build"
LOG_DIR = Path(__file__).resolve().parent / "logs"
STATIC_DIR = Path(__file__).resolve().parent / "static"
RESULTS_CSV = NNUE_CHESS_DIR / "training_results.csv"
STRENGTH_CSV = NNUE_CHESS_DIR / "strength_results.csv"
LOG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Job tracking
# ---------------------------------------------------------------------------

jobs = {}  # id -> dict (see start_job for shape)
jobs_lock = threading.Lock()
build_state = {"building": False, "log": None}
build_lock = threading.Lock()

PROGRESS_RE = re.compile(
    r"\[(?P<bar>[=> ]*)\]\s*(?P<pct>\d+)%\s*\((?P<cur>\d+)/(?P<tot>\d+)\)\s*"
    r"(?P<rate>[\d.]+)ms/item\s*ETA\s*(?P<eta>\d+)s(?:\s{2,}(?P<suffix>.*))?"
)


def collapse_progress_bar(text: str) -> str:
    """Collapses every burst of '\\r'-joined progress-bar updates (all the
    intermediate states of one bar, still literally present back-to-back in
    the raw log file) down to just the latest one per real line, so a log
    tail view isn't thousands of overwritten bar frames."""
    lines = text.split("\n")
    return "\n".join(line.split("\r")[-1] if "\r" in line else line for line in lines)


def read_progress(log_path: Path):
    try:
        raw = log_path.read_text(errors="replace")
    except FileNotFoundError:
        return None, ""
    clean = collapse_progress_bar(raw)
    matches = list(PROGRESS_RE.finditer(raw))
    progress = None
    if matches:
        m = matches[-1]
        progress = {
            "pct": int(m.group("pct")),
            "cur": int(m.group("cur")),
            "tot": int(m.group("tot")),
            "rate_ms": float(m.group("rate")),
            "eta_s": int(m.group("eta")),
            "suffix": (m.group("suffix") or "").strip(),
        }
    tail = "\n".join(clean.splitlines()[-200:])
    return progress, tail


def reap(job_id):
    """Runs in a background thread per job: waits for the process to exit,
    then finalizes its status. Keeps the request handlers themselves
    non-blocking - starting a job returns as soon as it's spawned."""
    with jobs_lock:
        proc = jobs[job_id]["proc"]
    returncode = proc.wait()
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            return
        job["returncode"] = returncode
        job["end_time"] = time.time()
        if job["status"] == "stopping":
            job["status"] = "stopped"
        else:
            job["status"] = "done" if returncode == 0 else "failed"


def start_job(tool, argv, threads, extra, cmd=None):
    job_id = uuid.uuid4().hex[:8]
    log_path = LOG_DIR / f"{job_id}.log"
    if cmd is None:
        binary = BUILD_DIR / tool
        if not binary.exists():
            raise RuntimeError(f"{binary} existiert nicht - erst bauen (siehe Architektur-Panel)")
        cmd = [str(binary)] + argv
    log_file = open(log_path, "wb", buffering=0)
    proc = subprocess.Popen(
        cmd,
        cwd=str(NNUE_CHESS_DIR),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # own process group -> can kill Stockfish children too
    )
    job = {
        "id": job_id,
        "tool": tool,
        "argv": argv,
        "threads": threads,
        "cmd_str": " ".join(cmd),
        "proc": proc,
        "log_path": str(log_path),
        "status": "running",
        "start_time": time.time(),
        "end_time": None,
        "returncode": None,
        "external": False,
        **extra,
    }
    with jobs_lock:
        jobs[job_id] = job
    threading.Thread(target=reap, args=(job_id,), daemon=True).start()
    return job_id


def stop_job(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.get("external"):
            pid = job["pid"]
        else:
            pid = job["proc"].pid
            job["status"] = "stopping"
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except ProcessLookupError:
        return

    def escalate():
        time.sleep(5)
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass

    threading.Thread(target=escalate, daemon=True).start()


def active_thread_sum():
    with jobs_lock:
        return sum(j.get("threads", 0) or 0 for j in jobs.values() if j["status"] == "running")


# ---------------------------------------------------------------------------
# External (non-GUI-launched) job discovery, e.g. a long full_cycle run
# started from a terminal before the GUI existed - surfaced read-only
# (no captured log file to tail, but still visible + stoppable).
# ---------------------------------------------------------------------------

def scan_external_jobs():
    try:
        out = subprocess.run(["pgrep", "-af", r"build/(full_cycle|self_play)"], capture_output=True, text=True)
    except FileNotFoundError:
        return
    known_pids = set()
    with jobs_lock:
        for j in jobs.values():
            if not j.get("external") and j["status"] == "running":
                known_pids.add(j["proc"].pid)
    for line in out.stdout.splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        pid_str, cmdline = parts
        pid = int(pid_str)
        if pid in known_pids or pid == os.getpid():
            continue
        job_id = f"ext-{pid}"
        with jobs_lock:
            if job_id in jobs:
                continue
            jobs[job_id] = {
                "id": job_id,
                "tool": "full_cycle" if "full_cycle" in cmdline else "self_play",
                "argv": [],
                "threads": 0,
                "cmd_str": cmdline,
                "pid": pid,
                "log_path": None,
                "status": "running",
                "start_time": None,
                "end_time": None,
                "returncode": None,
                "external": True,
            }


def external_reaper():
    while True:
        time.sleep(3)
        scan_external_jobs()
        with jobs_lock:
            for job in jobs.values():
                if job.get("external") and job["status"] == "running":
                    if not Path(f"/proc/{job['pid']}").exists():
                        job["status"] = "done"
                        job["end_time"] = time.time()


threading.Thread(target=external_reaper, daemon=True).start()


# ---------------------------------------------------------------------------
# Leaderboard (mirrors leaderboard.hpp's ranking_key: validation_loss when
# available - scale-corrected, comparable across stockfish/selfplay - else
# best_loss)
# ---------------------------------------------------------------------------

def load_leaderboard(top_n=25):
    if not RESULTS_CSV.exists():
        return []
    rows = []
    with open(RESULTS_CSV, newline="") as f:
        for r in csv.DictReader(f):
            try:
                val_loss = float(r["validation_loss"])
                best_loss = float(r["best_loss"])
                rows.append(
                    {
                        "name": r["name"],
                        "method": r["method"],
                        "timestamp": r["timestamp"],
                        "architecture": f"{r['acc']},{r['h1']},{r['h2']},{r['h3']}",
                        "depth": int(r["depth"]),
                        "samples": int(r["samples"]),
                        "games": int(r["games"]),
                        "best_loss": best_loss,
                        "final_loss": float(r["final_loss"]),
                        "total_time_ms": float(r["total_time_ms"]),
                        "validation_loss": val_loss,
                        "validation_correlation": float(r["validation_correlation"]),
                        "rank_key": val_loss if val_loss >= 0.0 else best_loss,
                    }
                )
            except (KeyError, ValueError):
                continue
    rows.sort(key=lambda r: r["rank_key"])
    return rows[:top_n]


def load_strength_results(top_n=25):
    if not STRENGTH_CSV.exists():
        return []
    rows = []
    with open(STRENGTH_CSV, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append(
                    {
                        "name": r["name"],
                        "net": r["net"],
                        "timestamp": r["timestamp"],
                        "estimated_elo": int(r["estimated_elo"]),
                        "bracket_lo": int(r["bracket_lo"]),
                        "bracket_hi": int(r["bracket_hi"]),
                        "rounds": int(r["rounds"]),
                        "total_games": int(r["total_games"]),
                    }
                )
            except (KeyError, ValueError):
                continue
    rows.sort(key=lambda r: -r["estimated_elo"])
    return rows[:top_n]


def piece_count(fen):
    return sum(1 for c in fen.split(" ", 1)[0] if c.isalpha())


def sample_dataset_lines(path, sample_size=20000):
    """Uniform-ish sample of a .dataset file's lines without reading the
    whole thing - these files run 500MB+/10M lines (see full_cycle.cpp's
    gen_data), so a full scan for an on-demand GUI chart would be
    noticeably slow. Small files (<5MB) are read in full instead, since
    random byte-offset seeking has more relative overhead there and isn't
    needed for correctness at that size."""
    size = path.stat().st_size
    if size < 5_000_000:
        with open(path, "r", errors="ignore") as f:
            lines = f.readlines()
        return lines if len(lines) <= sample_size else random.sample(lines, sample_size)

    rng = random.Random(42)
    lines = []
    with open(path, "rb") as f:
        for _ in range(sample_size):
            offset = rng.randint(0, size - 1)
            f.seek(offset)
            f.readline()  # discard the (likely partial) line we landed in the middle of
            line = f.readline()
            if line:
                lines.append(line.decode(errors="ignore"))
    return lines


def analyze_dataset(path):
    lines = sample_dataset_lines(path)
    pieces, cps, white = [], [], 0
    for line in lines:
        parts = line.strip().split(";")
        if len(parts) < 3:
            continue
        try:
            cp = int(parts[1])
        except ValueError:
            continue
        pieces.append(piece_count(parts[0]))
        cps.append(cp)
        if parts[2] == "w":
            white += 1
    n = len(pieces)
    if n == 0:
        return {"n_sampled": 0}

    histogram = [0] * 33  # piece count 0..32
    for p in pieces:
        histogram[max(0, min(32, p))] += 1
    phases = {"opening": 0, "midgame": 0, "endgame": 0, "deep_endgame": 0}
    for p in pieces:
        if p >= 28: phases["opening"] += 1
        elif p >= 16: phases["midgame"] += 1
        elif p >= 8: phases["endgame"] += 1
        else: phases["deep_endgame"] += 1
    cps.sort()

    def cp_pct(q):
        return cps[min(n - 1, int(q * n))]

    return {
        "n_sampled": n,
        "file_size_mb": round(path.stat().st_size / 1_000_000, 1),
        "piece_histogram": histogram,
        "phase_pct": {k: round(100.0 * v / n, 1) for k, v in phases.items()},
        "cp": {"min": cps[0], "p10": cp_pct(0.10), "p50": cp_pct(0.50), "p90": cp_pct(0.90), "max": cps[-1]},
        "white_pct": round(100.0 * white / n, 1),
    }


def build_arch():
    cache = BUILD_DIR / "CMakeCache.txt"
    if not cache.exists():
        return None
    vals = {}
    for line in cache.read_text().splitlines():
        for key in ("NNUE_ACC_SIZE", "NNUE_H1", "NNUE_H2", "NNUE_H3"):
            if line.startswith(key + ":"):
                vals[key] = line.split("=", 1)[1]
    if len(vals) != 4:
        return None
    return {
        "acc": vals["NNUE_ACC_SIZE"],
        "h1": vals["NNUE_H1"],
        "h2": vals["NNUE_H2"],
        "h3": vals["NNUE_H3"],
    }


def list_files(pattern):
    return sorted(p.name for p in NNUE_CHESS_DIR.glob(pattern))


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # keep stdout clean - the GUI's own log tailing is enough

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def _serve_static(self, path):
        if path == "/":
            path = "/index.html"
        fs_path = (STATIC_DIR / path.lstrip("/")).resolve()
        if STATIC_DIR not in fs_path.parents and fs_path != STATIC_DIR:
            self.send_error(403)
            return
        if not fs_path.is_file():
            self.send_error(404)
            return
        content_type = {
            ".html": "text/html",
            ".js": "application/javascript",
            ".css": "text/css",
        }.get(fs_path.suffix, "application/octet-stream")
        body = fs_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # -- GET ----------------------------------------------------------------
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            scan_external_jobs()
            with jobs_lock:
                job_list = []
                for j in jobs.values():
                    progress, tail = (None, "")
                    if j["log_path"]:
                        progress, tail = read_progress(Path(j["log_path"]))
                    job_list.append(
                        {
                            "id": j["id"],
                            "tool": j["tool"],
                            "cmd_str": j["cmd_str"],
                            "threads": j["threads"],
                            "status": j["status"],
                            "start_time": j["start_time"],
                            "end_time": j["end_time"],
                            "returncode": j["returncode"],
                            "external": j["external"],
                            "progress": progress,
                            "log_tail": tail,
                            "name": j.get("name", ""),
                            "mode": j.get("mode", ""),
                        }
                    )
            self._json(
                {
                    "cores": os.cpu_count() or 1,
                    "active_threads": active_thread_sum(),
                    "build_arch": build_arch(),
                    "building": build_state["building"],
                    "jobs": job_list,
                    "leaderboard": load_leaderboard(),
                    "strength_leaderboard": load_strength_results(),
                    "datasets": list_files("*.dataset"),
                    "networks": list_files("*.nnue"),
                    "loss_curves": [p[: -len("_loss.csv")] for p in list_files("*_loss.csv")],
                }
            )
            return
        if parsed.path == "/api/dataset_stats":
            qs = parse_qs(parsed.query)
            name = (qs.get("name") or [""])[0]
            if not re.match(r"^[A-Za-z0-9_.-]+$", name or ""):
                self._json({"error": "ungueltiger name"}, 400)
                return
            path = NNUE_CHESS_DIR / name
            if not path.exists():
                self._json({"error": f"{name} nicht gefunden"}, 404)
                return
            try:
                self._json(analyze_dataset(path))
            except Exception as e:  # noqa: BLE001
                self._json({"error": str(e)}, 500)
            return
        if parsed.path == "/api/loss":
            qs = parse_qs(parsed.query)
            name = (qs.get("name") or [""])[0]
            if not re.match(r"^[A-Za-z0-9_.-]+$", name or ""):
                self._json({"error": "ungueltiger name"}, 400)
                return
            csv_path = NNUE_CHESS_DIR / f"{name}_loss.csv"
            steps, losses_ = [], []
            if csv_path.exists():
                with open(csv_path, newline="") as f:
                    for row in csv.DictReader(f):
                        try:
                            steps.append(int(row["step"]))
                            losses_.append(float(row["loss"]))
                        except (KeyError, ValueError):
                            continue
            self._json({"steps": steps, "losses": losses_})
            return
        if parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/log"):
            job_id = parsed.path.split("/")[3]
            with jobs_lock:
                job = jobs.get(job_id)
            if job is None or not job["log_path"]:
                self._json({"error": "no log for this job"}, 404)
                return
            _, tail = read_progress(Path(job["log_path"]))
            self._json({"tail": tail})
            return
        self._serve_static(parsed.path)

    # -- POST -----------------------------------------------------------------
    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/jobs/start":
                body = self._read_json()
                job_id = self._handle_start(body)
                self._json({"job_id": job_id})
                return
            if parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/stop"):
                job_id = parsed.path.split("/")[3]
                stop_job(job_id)
                self._json({"ok": True})
                return
            if parsed.path == "/api/build":
                body = self._read_json()
                self._handle_build(body)
                self._json({"ok": True})
                return
        except Exception as e:  # noqa: BLE001 - surfaced to the GUI, not a crash
            self._json({"error": str(e)}, 400)
            return
        self.send_error(404)

    def _handle_start(self, body):
        if build_state["building"]:
            raise RuntimeError("ein Rebuild läuft gerade - bitte warten, bevor ein neues Training startet")
        tool = body.get("tool")
        name = (body.get("name") or "run").strip()
        if not re.match(r"^[A-Za-z0-9_.-]+$", name):
            raise RuntimeError("Name darf nur Buchstaben, Ziffern, _ . - enthalten")

        if tool == "full_cycle":
            mode = body.get("mode", "full")
            threads = int(body.get("threads") or 0)
            argv = ["--mode", mode, "--name", name, "--lr", str(body.get("lr", "1e-4")),
                    "--lr-half-life", str(body.get("lr_half_life", "100000"))]
            if mode in ("full", "gen_data"):
                argv += ["--depth", str(int(body.get("depth", 10))), "--samples", str(int(body.get("samples", 20000)))]
            if mode == "only_train":
                dataset = body.get("dataset", "")
                if not dataset:
                    raise RuntimeError("--dataset ist bei only_train erforderlich")
                argv += ["--dataset", dataset, "--epochs", str(int(body.get("epochs", 1)))]
                if body.get("batch_size"):
                    argv += ["--batch-size", str(int(body["batch_size"]))]
                piece_min, piece_max = int(body.get("piece_min", 2)), int(body.get("piece_max", 32))
                if piece_min > 2 or piece_max < 32:
                    argv += ["--piece-min", str(piece_min), "--piece-max", str(piece_max)]
            if threads > 0:
                argv += ["--threads", str(threads)]
            job_id = start_job("full_cycle", argv, threads, {"name": name, "mode": mode})
            return job_id

        if tool == "self_play":
            td_mode = body.get("td_mode", "leaf")
            argv = ["--games", str(int(body.get("games", 200))), "--td-mode", td_mode, "--name", name,
                     "--max-plies", str(int(body.get("max_plies", 150))),
                     "--sync-every", str(int(body.get("sync_every", 10))),
                     "--lr", str(body.get("lr", "5e-4")),
                     "--lr-half-life", str(body.get("lr_half_life", "100000"))]
            if td_mode == "leaf":
                argv += ["--search-depth", str(int(body.get("search_depth", 2)))]
            else:
                argv += ["--rollout-depth", str(int(body.get("rollout_depth", 7))),
                         "--td-lambda", str(body.get("td_lambda", 0.7))]
            if body.get("init_from"):
                argv += ["--init-from", body["init_from"]]
            # self_play has no internal --threads (one sequential game loop) -
            # count it as 1 core for the "cores in use" gauge, not 0.
            job_id = start_job("self_play", argv, 1, {"name": name, "mode": td_mode})
            return job_id

        if tool == "strength_search":
            threads = int(body.get("threads") or 0)
            bucket_rows = body.get("net_buckets")  # [{"min": N, "max": N, "net": "path.nnue"}, ...] or falsy
            if bucket_rows:
                for r in bucket_rows:
                    if not r.get("net"):
                        raise RuntimeError("jeder Bucket braucht ein Netz")
                net_buckets_arg = ",".join(f"{int(r['min'])}-{int(r['max'])}:{r['net']}" for r in bucket_rows)
                argv = ["--net-buckets", net_buckets_arg]
            else:
                net = body.get("net", "")
                if not net:
                    raise RuntimeError("--net oder mehrere Buckets sind erforderlich")
                argv = ["--net", net]
            argv += ["--name", name,
                    "--games-per-level", str(int(body.get("games_per_level", 30))),
                    "--movetime-ms", str(int(body.get("movetime_ms", 100))),
                    "--max-plies", str(int(body.get("max_plies", 200))),
                    "--initial-elo", str(int(body.get("initial_elo", 1500)))]
            if threads > 0:
                argv += ["--threads", str(threads)]
            if body.get("gen_dataset"):
                save_dataset = body.get("save_dataset") or f"{name}_from_play.dataset"
                argv += ["--save-dataset", save_dataset,
                         "--gen-dataset-games", str(int(body.get("gen_dataset_games", 100))),
                         "--dataset-depth", str(int(body.get("dataset_depth", 12)))]
            cmd = [sys.executable, str(Path(__file__).resolve().parent / "strength_search.py")] + argv
            job_id = start_job("strength_search", argv, threads, {"name": name, "mode": "strength"}, cmd=cmd)
            return job_id

        raise RuntimeError(f"unbekanntes tool: {tool}")

    def _handle_build(self, body):
        with jobs_lock:
            if any(j["status"] == "running" for j in jobs.values()):
                raise RuntimeError("es laufen noch Trainings - erst stoppen, dann neu bauen (gleiche build/ und Binaries)")
        acc, h1, h2, h3 = body["acc"], body["h1"], body["h2"], body["h3"]
        if not build_lock.acquire(blocking=False):
            raise RuntimeError("es läuft bereits ein Build")
        build_state["building"] = True

        def run_build():
            log_path = LOG_DIR / "build.log"
            try:
                with open(log_path, "w") as f:
                    subprocess.run(
                        ["cmake", "-S", ".", "-B", "build", "-DCMAKE_BUILD_TYPE=Release",
                         f"-DNNUE_ACC_SIZE={acc}", f"-DNNUE_H1={h1}", f"-DNNUE_H2={h2}", f"-DNNUE_H3={h3}"],
                        cwd=str(NNUE_CHESS_DIR), stdout=f, stderr=subprocess.STDOUT, check=True,
                    )
                    subprocess.run(
                        ["cmake", "--build", "build", "-j", "--target", "full_cycle", "self_play"],
                        cwd=str(NNUE_CHESS_DIR), stdout=f, stderr=subprocess.STDOUT, check=True,
                    )
            finally:
                build_state["building"] = False
                build_lock.release()

        threading.Thread(target=run_build, daemon=True).start()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    scan_external_jobs()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"nnue-chess training GUI: http://localhost:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
