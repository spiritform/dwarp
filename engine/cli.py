"""Render straight from a job JSON — handy for engine work without the UI.

    .venv\\Scripts\\python -m engine.cli job.json
"""
import json
import sys

from engine.warp import RenderJob, assemble, render


def main():
    job = RenderJob.from_dict(json.load(open(sys.argv[1], encoding="utf-8")))
    frames = render(job)
    if frames:
        fps = float(sys.argv[2]) if len(sys.argv) > 2 else 24 / job.nth
        print("video:", assemble(job.out_dir, fps))


if __name__ == "__main__":
    main()
