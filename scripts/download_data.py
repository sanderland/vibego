"""Download KataGo training data from the public archive.

Data lives at https://katagoarchive.org/kata1/trainingdata/ as daily .tgz archives
(e.g. 2021-03-01npzs.tgz), each bundling many .npz files of training rows (19x19).

Examples:
    uv run python scripts/download_data.py --n 5                 # 5 most recent days
    uv run python scripts/download_data.py --n 20 --from 2021-01-01
    uv run python scripts/download_data.py --list
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tarfile

import requests

BASE = "https://katagoarchive.org/kata1/trainingdata/"


def list_archives() -> list[str]:
    html = requests.get(BASE + "index.html", timeout=60).text
    names = re.findall(r'href="\./([^"]+npzs\.tgz)"', html)
    return sorted(names)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data")
    p.add_argument("--n", type=int, default=5, help="number of daily archives to fetch")
    p.add_argument("--from", dest="from_date", default=None,
                   help="earliest date YYYY-MM-DD; take the n archives starting here")
    p.add_argument("--recent", action="store_true",
                   help="take the n most recent instead of the n oldest")
    p.add_argument("--keep-tgz", action="store_true", help="keep the .tgz after extracting")
    p.add_argument("--list", action="store_true", help="just list available archives")
    args = p.parse_args()

    archives = list_archives()
    print(f"{len(archives)} daily archives available ({archives[0]} .. {archives[-1]})")
    if args.list:
        return

    if args.from_date:
        archives = [a for a in archives if a >= args.from_date]
    chosen = archives[-args.n:] if args.recent else archives[: args.n]
    os.makedirs(args.out, exist_ok=True)
    print(f"downloading {len(chosen)} archives -> {args.out}/")

    for name in chosen:
        tgz_path = os.path.join(args.out, name)
        if not os.path.exists(tgz_path):
            url = BASE + name
            print(f"  GET {url}")
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(tgz_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
        subdir = os.path.join(args.out, name.replace("npzs.tgz", ""))
        os.makedirs(subdir, exist_ok=True)
        with tarfile.open(tgz_path) as tar:
            members = [m for m in tar.getmembers() if m.name.endswith(".npz")]
            tar.extractall(subdir, members=members)
        n_npz = len([f for f in os.listdir(subdir) if f.endswith(".npz")])
        print(f"  {name}: extracted {n_npz} npz")
        if not args.keep_tgz:
            os.remove(tgz_path)

    from nanogo.net import data as _data
    total = len(_data.list_npz(args.out))
    print(f"done. {total} npz files under {args.out}/")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
