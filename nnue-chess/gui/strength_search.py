#!/usr/bin/env python3
"""Adaptive search for "what Stockfish Elo does this net's TestEngine play
at" - repeatedly calls the C++ strength_match binary (plays --games-per-
level games at ONE fixed Stockfish UCI_Elo, all --threads used, parallel
games - see testing/strength_match.cpp) at different Elo values, first
taking large steps to bracket the point where the TestEngine's win rate
crosses 50%, then bisecting within that bracket down to a small step size.

Why this logic lives in Python and not in strength_match.cpp itself: that
binary's job is "play N games at one fixed strength", which is independently
useful/testable and keeps the parallel-game-worker C++ code simple. The
meta-algorithm on top - and the live progress reporting the GUI already
knows how to parse (see gui/server.py's PROGRESS_RE, matching
training/progress_bar.hpp's format exactly) - fits naturally as the outer
loop here, one subprocess call per tested Elo level.

Usage (normally launched by the GUI, but works standalone):
    python3 gui/strength_search.py --net my_net.nnue --games-per-level 40 \
        --threads 8 --name my_net_strength
"""
import argparse
import csv
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

NNUE_CHESS_DIR = Path(__file__).resolve().parent.parent
BUILD_DIR = NNUE_CHESS_DIR / "build"
RESULTS_CSV = NNUE_CHESS_DIR / "strength_results.csv"

RESULT_RE = re.compile(r"RESULT wins=(\d+) draws=(\d+) losses=(\d+) games=(\d+) score=([\d.]+) elo=(\d+)")
DEPTH_RE = re.compile(
    r"DEPTH win_avg_depth=([\d.]+) win_avg_plies=([\d.]+) draw_avg_depth=([\d.]+) draw_avg_plies=([\d.]+) "
    r"loss_avg_depth=([\d.]+) loss_avg_plies=([\d.]+)"
)

MIN_ELO = 1320  # Stockfish's own UCI_Elo floor (roughly - it clamps internally regardless)
MAX_ELO = 3190


def print_progress_bar(current, total, rate_ms_per_item, eta_s, suffix=""):
    """Mirrors training/progress_bar.hpp's exact output format so the GUI's
    existing regex (server.py's PROGRESS_RE) parses this unmodified."""
    width = 30
    frac = current / total if total > 0 else 1.0
    filled = int(frac * width)
    bar = "=" * filled + (">" if filled < width else "") + " " * max(0, width - filled - (1 if filled < width else 0))
    sys.stdout.write(f"\r[{bar}] {int(frac * 100):3d}%  ({current}/{total})  {rate_ms_per_item:.1f}ms/item  "
                      f"ETA {eta_s:.0f}s{'  ' if suffix else ''}{suffix}")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")


def run_level(net, net_buckets, elo, games, threads, movetime_ms, max_plies, round_no, save_dataset=None,
              dataset_depth=12, label="Runde"):
    print(f"\n== {label} {round_no}: teste Elo {elo} ({games} Spiele, {threads} Threads) ==")
    sys.stdout.flush()
    net_args = ["--net-buckets", net_buckets] if net_buckets else ["--net", net]
    cmd = [str(BUILD_DIR / "strength_match")] + net_args + [
           "--elo", str(elo), "--games", str(games),
           "--threads", str(threads), "--movetime-ms", str(movetime_ms), "--max-plies", str(max_plies)]
    if save_dataset:
        cmd += ["--save-dataset", save_dataset, "--dataset-depth", str(dataset_depth)]
    proc = subprocess.Popen(cmd, cwd=str(NNUE_CHESS_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
    buf = b""
    while True:
        chunk = proc.stdout.read(4096)
        if not chunk:
            break
        buf += chunk
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
    proc.wait()
    text = buf.decode(errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"strength_match exited with code {proc.returncode} at elo {elo}:\n{text[-2000:]}")
    m = RESULT_RE.search(text)
    if not m:
        raise RuntimeError(f"could not find a RESULT line in strength_match's output at elo {elo}")
    wins, draws, losses, n, score, _ = m.groups()
    result = {"wins": int(wins), "draws": int(draws), "losses": int(losses), "games": int(n), "score": float(score)}
    dm = DEPTH_RE.search(text)
    if dm:
        wd, wp, dd, dp, ld, lp = (float(x) for x in dm.groups())
        result.update(win_avg_depth=wd, win_avg_plies=wp, draw_avg_depth=dd, draw_avg_plies=dp,
                      loss_avg_depth=ld, loss_avg_plies=lp)
        print(f"  Tiefe: Siege ø{wd:.1f} ({wp:.0f} Halbzüge)  Remis ø{dd:.1f} ({dp:.0f})  "
              f"Niederlagen ø{ld:.1f} ({lp:.0f})")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default=None, help="single net used for every position")
    ap.add_argument("--net-buckets", default=None,
                     help="several nets, each specialized on one piece-count range - "
                          "\"MIN-MAX:PATH,MIN-MAX:PATH,...\" - see strength_match --help. "
                          "Overrides --net if both are given.")
    ap.add_argument("--games-per-level", type=int, default=30)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--movetime-ms", type=int, default=100)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--initial-elo", type=int, default=1500)
    ap.add_argument("--initial-step", type=int, default=400)
    ap.add_argument("--min-step", type=int, default=20)
    ap.add_argument("--max-rounds", type=int, default=14)
    ap.add_argument("--name", default="strength_run")
    ap.add_argument("--save-dataset", default=None,
                     help="after the Elo is determined, play --gen-dataset-games more games at exactly that "
                          "Elo and append every position the TestEngine faced (fen;stockfish_cp;stm) to this "
                          "file - real, comparably-matched-opponent positions to train on afterwards")
    ap.add_argument("--gen-dataset-games", type=int, default=100)
    ap.add_argument("--dataset-depth", type=int, default=12)
    args = ap.parse_args()
    if not args.net and not args.net_buckets:
        ap.error("--net or --net-buckets is required")

    import os
    threads = args.threads if args.threads > 0 else (os.cpu_count() or 1)

    start_time = time.time()
    tested = {}  # elo -> result dict
    round_no = 0

    def test(elo):
        nonlocal round_no
        elo = max(MIN_ELO, min(MAX_ELO, elo))
        if elo in tested:
            return elo, tested[elo]
        round_no += 1
        r = run_level(args.net, args.net_buckets, elo, args.games_per_level, threads, args.movetime_ms,
                      args.max_plies, round_no)
        tested[elo] = r
        return elo, r

    # --- Phase 1: bracket the 50%-win-rate crossing with constant-size steps
    prev_elo, prev = test(args.initial_elo)
    direction = 1 if prev["score"] > 0.5 else -1
    step = args.initial_step

    bracket = None
    while round_no < args.max_rounds:
        next_elo_raw = prev_elo + direction * step
        clamped = max(MIN_ELO, min(MAX_ELO, next_elo_raw))
        if clamped == prev_elo:
            # already at the Elo boundary and still on the same side of 50% -
            # report the boundary itself as the best available estimate.
            bracket = (prev_elo, prev_elo)
            break
        next_elo, nxt = test(clamped)

        if (nxt["score"] - 0.5) * (prev["score"] - 0.5) <= 0:
            bracket = (prev_elo, next_elo) if prev_elo < next_elo else (next_elo, prev_elo)
            break
        prev_elo, prev = next_elo, nxt

    if bracket is None:
        bracket = (prev_elo, prev_elo)

    # --- Phase 2: bisect within the bracket -----------------------------
    lo, hi = bracket
    while (hi - lo) > args.min_step and round_no < args.max_rounds:
        mid = (lo + hi) // 2
        mid_elo, mid_res = test(mid)
        lo_res = tested.get(lo)
        if lo_res is not None and (lo_res["score"] - 0.5) * (mid_res["score"] - 0.5) <= 0:
            hi = mid_elo
        else:
            lo = mid_elo

    estimated_elo = (lo + hi) // 2
    total_games = sum(r["games"] for r in tested.values())
    total_time_s = time.time() - start_time

    print(f"\n== Spielstärke ermittelt: {args.name} ==")
    print(f"geschätztes Elo: {estimated_elo}")
    print(f"finale Bracket: [{lo}, {hi}]")
    print(f"Runden: {round_no}, Gesamtspiele: {total_games}, Zeit: {total_time_s:.0f}s")
    for elo in sorted(tested):
        r = tested[elo]
        print(f"  Elo {elo}: {r['wins']}-{r['draws']}-{r['losses']} (score {r['score']:.3f})")

    if args.save_dataset:
        print(f"\n== Datensatz-Generierung: {args.gen_dataset_games} Spiele bei geschätztem Elo {estimated_elo} ==")
        gen_result = run_level(args.net, args.net_buckets, estimated_elo, args.gen_dataset_games, threads,
                                args.movetime_ms, args.max_plies, 1, save_dataset=args.save_dataset,
                                dataset_depth=args.dataset_depth, label="Datensatz-Lauf")
        print(f"\n{gen_result['games']} Spiele gespielt (W{gen_result['wins']} D{gen_result['draws']} "
              f"L{gen_result['losses']}) - Positionen an {args.save_dataset} angehängt.")

    need_header = not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(["name", "net", "timestamp", "estimated_elo", "bracket_lo", "bracket_hi", "rounds",
                        "total_games", "total_time_s"])
        net_label = args.net_buckets if args.net_buckets else args.net
        w.writerow([args.name, net_label, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), estimated_elo, lo, hi,
                    round_no, total_games, f"{total_time_s:.0f}"])
    print(f"\nErgebnis gespeichert in {RESULTS_CSV.name}")


if __name__ == "__main__":
    main()
