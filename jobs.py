"""One render at a time, in a worker thread, with live snapshots for the page (SSE)."""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

TERMINAL = {"completed", "cancelled", "failed"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Job:
    def __init__(self, kind: str, run_id: str, payload: dict):
        self.id = uuid.uuid4().hex
        self.kind = kind                 # render | video
        self.run_id = run_id
        self.payload = payload
        self.state = self.stage = "queued"
        self.message = "waiting for the renderer"
        self.frame = self.total_frames = 0
        self.progress = 0.0
        self.logs: list[str] = []
        self.error = self.traceback = None
        self.cancel_requested = False
        self.created_at, self.started_at, self.finished_at = _now(), None, None
        self.revision = 0

    def snapshot(self) -> dict:
        return {k: getattr(self, k) for k in (
            "id", "kind", "run_id", "state", "stage", "message", "frame", "total_frames", "progress",
            "logs", "error", "traceback", "cancel_requested", "created_at", "started_at", "finished_at",
            "revision")} | {"logs": list(self.logs)}


class JobManager:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._queue: list[Job] = []
        self._cond = threading.Condition()
        threading.Thread(target=self._loop, daemon=True).start()

    # -------------------------------------------------------------- public
    def submit(self, kind: str, run_id: str, payload: dict) -> Job:
        job = Job(kind, run_id, payload)
        with self._cond:
            self._jobs[job.id] = job
            self._queue.append(job)
            self._cond.notify_all()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[dict]:
        return [j.snapshot() for j in sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)]

    def busy(self) -> bool:
        return any(j.state not in TERMINAL for j in self._jobs.values())

    def cancel(self, job_id: str) -> Job | None:
        job = self._jobs.get(job_id)
        if job:
            with self._cond:
                job.cancel_requested = True
                if job.state == "queued":
                    job.state = job.stage = "cancelled"
                    job.message = "cancelled before starting"
                self._touch(job)
        return job

    def wait_for_change(self, job: Job, revision: int, timeout: float = 15.0) -> dict:
        with self._cond:
            self._cond.wait_for(lambda: job.revision != revision or job.state in TERMINAL, timeout=timeout)
            return job.snapshot()

    # -------------------------------------------------------------- worker
    def _touch(self, job: Job):
        with self._cond:          # re-entrant (Condition wraps an RLock), safe from any caller
            job.revision += 1
            self._cond.notify_all()

    def _log(self, job: Job, line: str):
        with self._cond:
            stamp = time.strftime("%H:%M:%S")
            job.logs.append(f"[{stamp}] {line}")
            job.logs[:] = job.logs[-500:]
            self._touch(job)
        print(f"[{job.run_id}] {line}", flush=True)

    def _loop(self):
        while True:
            with self._cond:
                self._cond.wait_for(lambda: any(j.state == "queued" for j in self._queue))
                job = next(j for j in self._queue if j.state == "queued")
                self._queue.remove(job)
                job.state, job.stage, job.started_at = "running", "starting", _now()
                self._touch(job)
            try:
                self._run(job)
                with self._cond:
                    if job.cancel_requested:
                        job.state = job.stage = "cancelled"
                        job.message = f"cancelled at frame {job.frame}"
                    else:
                        job.state = job.stage = "completed"
                        job.progress = 100.0
                        job.message = "done"
            except Exception as exc:  # noqa: BLE001 — surface everything to the page
                with self._cond:
                    job.state = job.stage = "failed"
                    job.error = f"{type(exc).__name__}: {exc}"
                    job.traceback = traceback.format_exc()
                    job.message = "failed"
                print(job.traceback, flush=True)
            finally:
                with self._cond:
                    job.finished_at = _now()
                    self._touch(job)

    def _run(self, job: Job):
        from engine.warp import RenderJob, assemble, render
        import runs

        run_path = runs.run_dir(job.run_id)
        if job.kind == "video":
            job.stage, job.message = "assembling", "assembling mp4"
            self._touch(job)
            assemble(str(run_path), job.payload["fps"])
            self._log(job, "assembled video.mp4")
            return

        rj = RenderJob.from_dict({**job.payload["job"], "out_dir": str(run_path)})

        def progress(stage, frame, total, message):
            with self._cond:
                job.stage, job.message = stage, message
                job.frame, job.total_frames = frame, total
                job.progress = frame / total * 100 if total else 0.0
                self._touch(job)

        t0 = time.time()
        written = render(rj, progress=progress, cancelled=lambda: job.cancel_requested,
                         log=lambda line: self._log(job, line))
        if written and not job.cancel_requested:
            job.stage, job.message = "assembling", "assembling mp4"
            self._touch(job)
            assemble(str(run_path), job.payload.get("fps", 24))
            self._log(job, f"assembled video.mp4 ({len(written)} frames)")
        took = time.time() - t0
        m, sec = divmod(round(took), 60)
        h, m = divmod(m, 60)
        per = f", {took / len(written):.1f}s/frame" if written else ""
        self._log(job, f"total {f'{h}h ' if h else ''}{m}m {sec:02d}s{per}")
