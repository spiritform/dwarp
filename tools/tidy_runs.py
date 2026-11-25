"""Move the run folders already in renders/ into folders for the day they were made.

    .venv\\Scripts\\python tools\\tidy_runs.py --dry     # what it would do
    .venv\\Scripts\\python tools\\tidy_runs.py           # do it

renders/0 .. renders/1050 becomes renders/20260923/0 .. renders/20261125/1050. A run keeps its
number, which is its id, so nothing that points at a run -- a label, a source_run, a link you had
open -- has to change, and DWARP finds a run in either place. New runs go under today's date on
their own; this is only for the ones made before that.

A move is a rename inside the same folder, so it costs nothing and moves no bytes. Run it with
nothing rendering: moving a folder the engine is writing into would lose the render.
"""
from __future__ import annotations

import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import runs                                                    # noqa: E402


def made_at(path: Path) -> float:
    """When the run's frames were written -- the folder's own time moves with thumbnails and hiding."""
    for name in ("frames", "meta.json", "job.json"):
        f = path / name
        try:
            return f.stat().st_mtime
        except OSError:
            continue
    return path.stat().st_mtime


def main(dry: bool) -> int:
    root = runs.RENDERS
    if not root.is_dir():
        print(f"no renders folder at {root}")
        return 1
    flat = []
    with os.scandir(root) as it:
        for e in it:
            if not e.is_dir() or not e.name.isdigit():
                continue
            path = Path(e.path)
            if runs._DAY.match(e.name) and runs._is_day(path):   # already a day folder
                continue
            flat.append(path)
    if not flat:
        print("nothing to tidy -- every run is already under a day")
        return 0

    flat.sort(key=lambda p: int(p.name))
    days: Counter[str] = Counter()
    moved = skipped = 0
    for path in flat:
        day = time.strftime("%Y%m%d", time.localtime(made_at(path)))
        dest = root / day / path.name
        if dest.exists():
            print(f"  ! {path.name} -> {day}/ already there, left alone")
            skipped += 1
            continue
        days[day] += 1
        if dry:
            continue
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.rename(path, dest)
            moved += 1
        except OSError as e:
            print(f"  ! {path.name} would not move: {e}")
            skipped += 1

    for day in sorted(days):
        print(f"  {day}  {days[day]} runs")
    print(f"\n{len(flat)} runs over {len(days)} days -- "
          + (f"nothing moved (--dry)" if dry else f"{moved} moved, {skipped} left alone"))
    if not dry:
        runs.index(fresh=True)
        print(f"{len(runs.index())} runs found after the move")
    return 0


if __name__ == "__main__":
    raise SystemExit(main("--dry" in sys.argv[1:]))
