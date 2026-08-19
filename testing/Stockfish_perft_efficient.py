#!/usr/bin/env python3
"""
Lädt Stockfish (neueste Version von GitHub) herunter, falls noch nicht
vorhanden, und führt den UCI-Befehl 'go perft <depth>' auf einer
beliebigen FEN-Position aus.

Nutzung:
    python stockfish_perft.py
    python stockfish_perft.py --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" --depth 5
    python stockfish_perft.py --fen "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1" --depth 4
"""

import os
import sys
import platform
import subprocess
import tarfile
import zipfile
import stat
import argparse
import urllib.request
import json

GITHUB_API_LATEST = "https://api.github.com/repos/official-stockfish/Stockfish/releases/latest"
INSTALL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stockfish_bin")

# Falls die GitHub-API mal ein Rate-Limit wirft, wird auf dieses (bekannt
# funktionierende) Release als Fallback zurückgegriffen.
FALLBACK_TAG = "sf_17.1"

# Standard: Schachbrett-Startposition
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def get_platform_asset_keywords():
    """Ermittelt, welche Schlüsselwörter das passende Release-Asset enthalten muss."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "windows":
        os_key, ext = "windows", ".zip"
    elif system == "darwin":
        os_key, ext = "macos", ".tar"
    else:
        os_key, ext = "ubuntu", ".tar"

    if machine in ("x86_64", "amd64"):
        arch_key = "x86-64"
    elif machine in ("arm64", "aarch64"):
        arch_key = "arm64"
    else:
        arch_key = machine

    return os_key, arch_key, ext


def find_release_asset():
    print("Suche nach der neuesten Stockfish-Version auf GitHub...")
    req = urllib.request.Request(GITHUB_API_LATEST, headers={"User-Agent": "python-perft-script"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)

    os_key, arch_key, ext = get_platform_asset_keywords()
    assets = data.get("assets", [])

    candidates = [
        a for a in assets
        if os_key in a["name"].lower()
        and arch_key in a["name"].lower()
        and a["name"].lower().endswith(ext)
    ]

    # AVX2-Variante bevorzugen (schneller), sonst irgendeine passende nehmen
    preferred = [a for a in candidates if "avx2" in a["name"].lower()]
    chosen = preferred[0] if preferred else (candidates[0] if candidates else None)

    if not chosen:
        raise RuntimeError(
            f"Kein passendes Release-Asset gefunden (os={os_key}, arch={arch_key}). "
            f"Verfügbare Assets: {[a['name'] for a in assets]}"
        )

    print(f"Gefunden: {chosen['name']} ({data.get('tag_name')})")
    return chosen["name"], chosen["browser_download_url"]


def fallback_release_asset():
    """Wird genutzt, wenn die GitHub-API nicht erreichbar ist (z.B. Rate-Limit)."""
    os_key, arch_key, ext = get_platform_asset_keywords()
    name = f"stockfish-{os_key}-{arch_key}-avx2{ext}"
    url = f"https://github.com/official-stockfish/Stockfish/releases/download/{FALLBACK_TAG}/{name}"
    print(f"GitHub-API nicht verfügbar, nutze Fallback-Release {FALLBACK_TAG}: {name}")
    return name, url


def download_file(url, dest_path):
    print(f"Lade herunter: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "python-perft-script"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest_path, "wb") as out:
        out.write(resp.read())
    print(f"Gespeichert unter: {dest_path}")


def extract_archive(archive_path, extract_to):
    print("Entpacke Archiv...")
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(extract_to)
    elif archive_path.endswith(".tar"):
        with tarfile.open(archive_path, "r") as t:
            t.extractall(extract_to)
    else:
        raise RuntimeError(f"Unbekanntes Archivformat: {archive_path}")


def find_stockfish_binary(search_dir):
    is_windows = platform.system().lower() == "windows"
    for root, _, files in os.walk(search_dir):
        for f in files:
            fl = f.lower()
            if not fl.startswith("stockfish"):
                continue
            if is_windows and fl.endswith(".exe"):
                return os.path.join(root, f)
            if not is_windows and "." not in fl:
                return os.path.join(root, f)
    return None


def ensure_stockfish():
    os.makedirs(INSTALL_DIR, exist_ok=True)

    existing = find_stockfish_binary(INSTALL_DIR)
    if existing:
        print(f"Stockfish bereits vorhanden: {existing}")
        return existing

    try:
        asset_name, url = find_release_asset()
    except Exception as exc:
        print(f"Hinweis: Abfrage der GitHub-API fehlgeschlagen ({exc}).")
        asset_name, url = fallback_release_asset()

    archive_path = os.path.join(INSTALL_DIR, asset_name)
    download_file(url, archive_path)
    extract_archive(archive_path, INSTALL_DIR)

    binary = find_stockfish_binary(INSTALL_DIR)
    if not binary:
        raise RuntimeError("Stockfish-Binary wurde nach dem Entpacken nicht gefunden.")

    if platform.system().lower() != "windows":
        st = os.stat(binary)
        os.chmod(binary, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    return binary


def run_perft(engine_path, fen, depth):
    commands = [
        "uci",
        "isready",
        f"position fen {fen}",
        f"go perft {depth}",
    ]

    proc = subprocess.Popen(
        [engine_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    input_str = "\n".join(commands) + "\nquit\n"
    output, _ = proc.communicate(input_str, timeout=180)
    return output


def main():
    parser = argparse.ArgumentParser(description="Stockfish herunterladen und Perft ausführen.")
    parser.add_argument("--fen", default=START_FEN,
                         help="FEN-Position (Standard: Startposition)")
    parser.add_argument("--depth", type=int, default=5, help="Perft-Tiefe (Standard: 5)")
    args = parser.parse_args()

    engine_path = ensure_stockfish()
    print(f"\nFühre 'go perft {args.depth}' aus für FEN:\n  {args.fen}\n")

    output = run_perft(engine_path, args.fen, args.depth)
    print(output)


if __name__ == "__main__":
    main()