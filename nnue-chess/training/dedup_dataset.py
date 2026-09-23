#!/usr/bin/env python3
"""Caps exact-duplicate positions in a generated .dataset file.

Streams the input line by line (safe on multi-GB files) and drops any line
whose FEN has already been kept --max-duplicates times. Tracks seen FENs via
hash() instead of the raw string to keep memory bounded on very large files.

Positions with very few pieces (e.g. 2 - just the two kings) have a small,
inherently bounded space of distinct legal placements; capping duplicates
there just means "keep at most N copies of each", it can't manufacture more
underlying diversity than actually exists on the board.
"""
import argparse
import collections
import sys


def piece_count(fen: str) -> int:
    placement = fen.split(" ", 1)[0]
    return sum(1 for c in placement if c.isalpha())


def dedup_key(fen: str) -> str:
    # First 4 FEN fields: piece placement, side to move, castling rights,
    # en-passant square - these are what actually distinguishes one
    # position from another for evaluation purposes. Deliberately drops
    # the halfmove-clock/fullmove-number fields: two lines can be the same
    # board+side-to-move reached via walks of different lengths, which
    # gives them different counters but makes them the same training
    # example in every way that matters - keying on the full FEN string
    # (including those counters) undercounts duplicates for exactly that
    # reason.
    fields = fen.split(" ")
    return " ".join(fields[:4])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="dataset file to clean (fen;cp;... per line)")
    ap.add_argument("-o", "--output", help="output path (default: <input>.deduped)")
    ap.add_argument("--max-duplicates", type=int, default=3,
                     help="keep at most this many copies of an identical FEN (default: 3)")
    ap.add_argument("--dry-run", action="store_true", help="only print before/after stats, write nothing")
    args = ap.parse_args()

    output_path = args.output or (args.input + ".deduped")

    seen = collections.Counter()
    before_hist = collections.Counter()
    after_hist = collections.Counter()
    total_in = 0
    total_out = 0

    out = None if args.dry_run else open(output_path, "w")
    try:
        with open(args.input) as f:
            for line in f:
                stripped = line.rstrip("\n")
                if not stripped:
                    continue
                total_in += 1
                fen = stripped.split(";", 1)[0]
                pc = piece_count(fen)
                before_hist[pc] += 1

                key = hash(dedup_key(fen))
                seen[key] += 1
                if seen[key] > args.max_duplicates:
                    continue

                total_out += 1
                after_hist[pc] += 1
                if out is not None:
                    out.write(line if line.endswith("\n") else line + "\n")
    finally:
        if out is not None:
            out.close()

    print(f"input:  {args.input}  ({total_in} lines)")
    if not args.dry_run:
        print(f"output: {output_path}  ({total_out} lines)")
    print(f"dropped {total_in - total_out} lines ({100 * (total_in - total_out) / max(1, total_in):.1f}%) "
          f"as duplicates beyond --max-duplicates={args.max_duplicates}\n")

    print(f"{'pieces':>7} {'before':>10} {'after':>10} {'dropped':>10}")
    for pc in sorted(before_hist):
        b = before_hist[pc]
        a = after_hist.get(pc, 0)
        print(f"{pc:>7} {b:>10} {a:>10} {b - a:>10}")

    if args.dry_run:
        print("\n(dry run - no output file written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
