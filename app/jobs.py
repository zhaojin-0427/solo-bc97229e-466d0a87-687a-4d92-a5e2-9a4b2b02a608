"""Job persistence (SQLite), the single background worker, and the shared
archive store's reference counting / garbage collection.

Status lifecycle: QUEUED -> RUNNING -> SUCCEEDED / FAILED.
On application start-up every non-terminal job is marked FAILED, so a restart
can never expose a SUCCEEDED row without a complete, atomically published file.

Storage backends
----------------
* ``shared`` (new uploads): ``files`` is a JSON list of
  ``{"sha256", "name", "size"}``; the bytes live once in the content-addressed
  store and the ``archives`` table counts how many jobs reference each digest.
* ``legacy`` (jobs created before the shared store existed): ``files`` is a
  JSON list of plain filesystem paths under the per-job upload directory.
  They keep working exactly as before, including extraction and download.

Reference counting is transactional with job creation/deletion: a committed
job always has its archives pinned (refs >= 1), and anything unreferenced —
a published upload whose job commit failed, an archive released by a deleted
job, a temp file left by a crashed upload — is reclaimed by ``gc_store``.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import config, store
from .engine import ExtractionError, Selection, run_extraction

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    params      TEXT NOT NULL,
    files       TEXT NOT NULL,
    error       TEXT,
    progress    INTEGER NOT NULL DEFAULT 0,
    total       INTEGER NOT NULL DEFAULT 0,
    result      TEXT
);

CREATE TABLE IF NOT EXISTS archives (
    sha256      TEXT PRIMARY KEY,
    path        TEXT NOT NULL,
    size        INTEGER NOT NULL,
    created_at  TEXT NOT NULL,
    refs        INTEGER NOT NULL DEFAULT 0
);
"""

TERMINAL_STATES = ("SUCCEEDED", "FAILED")

_lock = threading.Lock()
_start_lock = threading.Lock()
# netCDF4/HDF5 is not thread-safe: exactly one extraction may run in this
# process at a time, no matter how many worker threads exist.
_extraction_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None
_worker_started = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        config.DATA_HOME.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(config.DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.executescript(SCHEMA)
        # Migrate pre-shared-store databases: existing jobs keep their
        # per-job upload directories ("legacy" storage).
        cols = {r[1] for r in _conn.execute("PRAGMA table_info(jobs)")}
        if "storage" not in cols:
            _conn.execute(
                "ALTER TABLE jobs ADD COLUMN storage "
                "TEXT NOT NULL DEFAULT 'legacy'")
        _conn.commit()
    return _conn


# ---------------------------------------------------------------------------
# Shared-store garbage collection
# ---------------------------------------------------------------------------

def gc_store() -> dict:
    """Reclaim everything in the shared store that nothing references.

    * ``archives`` rows with ``refs <= 0`` (released by deleted jobs, or left
      behind when a job commit failed) — unless an upload in this process is
      currently reusing that digest;
    * store files with no ``archives`` row at all (crash between publish and
      commit) — again skipping in-flight digests;
    * stale ``.incoming-*`` temp files from interrupted uploads.

    The in-flight snapshot from ``store`` makes this safe to run at any time:
    a concurrent upload's digest/temp path is never reclaimed from under it.
    """
    inflight = store.inflight_snapshot()
    db = get_db()
    removed = {"rows": 0, "files": 0, "bytes": 0}

    with _lock:
        rows = db.execute(
            "SELECT sha256, path, size FROM archives WHERE refs <= 0"
        ).fetchall()
        victims = []
        for r in rows:
            if r["sha256"] in inflight["shas"]:
                continue  # a concurrent upload is about to reference it
            cur = db.execute(
                "DELETE FROM archives WHERE sha256=? AND refs <= 0",
                (r["sha256"],))
            if cur.rowcount:
                victims.append(r)
                removed["rows"] += 1
                removed["bytes"] += r["size"]
        db.commit()
        known = {row[0] for row in db.execute("SELECT path FROM archives")}

    for r in victims:
        try:
            os.remove(r["path"])
            removed["files"] += 1
        except OSError:
            pass

    # On-disk orphans: files in the store with no archives row.
    if config.STORE_DIR.is_dir():
        for p in config.STORE_DIR.rglob("*.nc"):
            if str(p) in known or p.stem in inflight["shas"]:
                continue
            try:
                removed["bytes"] += p.stat().st_size
                p.unlink()
                removed["files"] += 1
            except OSError:
                pass
        # Drop shard directories left empty.
        for d in sorted((d for d in config.STORE_DIR.rglob("*")
                         if d.is_dir()), key=lambda d: -len(d.parts)):
            try:
                d.rmdir()
            except OSError:
                pass

    # Stale temp files from interrupted uploads.
    for p in config.TMP_DIR.glob(".incoming-*"):
        if str(p) in inflight["temps"]:
            continue
        try:
            p.unlink()
        except OSError:
            pass

    return removed


def store_stats() -> dict:
    """Number of shared archives and the actual bytes they occupy."""
    db = get_db()
    with _lock:
        row = db.execute(
            "SELECT COUNT(*) AS c, COALESCE(SUM(size), 0) AS b "
            "FROM archives").fetchone()
    return {"count": row["c"], "bytes": row["b"]}


# ---------------------------------------------------------------------------
# Startup recovery
# ---------------------------------------------------------------------------

def recover_on_startup() -> None:
    """Recover from a previous process:

    * every QUEUED/RUNNING job is marked FAILED (its partial output, if any,
      is removed) — a restart never exposes a result that was not fully
      published;
    * published outputs whose job is gone or not SUCCEEDED are removed;
    * legacy upload directories no longer referenced by any job are removed;
    * the shared store is garbage-collected, so unreferenced leftovers are
      reclaimed while every accepted job's inputs stay complete and readable
      (they are pinned by refs >= 1).
    """
    db = get_db()
    with _lock:
        rows = db.execute(
            "SELECT id, result FROM jobs "
            "WHERE status IN ('QUEUED','RUNNING')").fetchall()
        for r in rows:
            db.execute(
                "UPDATE jobs SET status='FAILED', updated_at=?, "
                "error=? WHERE id=?",
                (_now(), "服务重启导致任务中断，已标记为失败", r["id"]))
            if r["result"]:
                p = Path(r["result"])
                # A RUNNING job never published via os.replace; remove stale
                # files defensively.
                if p.exists() and p.parent == config.OUTPUT_DIR:
                    try:
                        p.unlink()
                    except OSError:
                        pass
        db.commit()
        keep_results = {row[0] for row in db.execute(
            "SELECT result FROM jobs "
            "WHERE status='SUCCEEDED' AND result IS NOT NULL")}
        all_jobs = db.execute("SELECT files, storage FROM jobs").fetchall()
    for partial in config.TMP_DIR.glob(".partial-*"):
        try:
            partial.unlink()
        except OSError:
            pass
    # Orphaned published outputs (e.g. crash while deleting a job).
    for out in config.OUTPUT_DIR.glob("subset_*.nc"):
        if str(out) not in keep_results:
            try:
                out.unlink()
            except OSError:
                pass
    # Orphaned legacy per-job upload directories.
    referenced_dirs = set()
    for r in all_jobs:
        try:
            files = json.loads(r["files"])
        except (ValueError, TypeError):
            continue
        for item in files:
            if isinstance(item, str):
                p = Path(item)
                if p.parent.parent == config.UPLOAD_DIR:
                    referenced_dirs.add(p.parent)
    if config.UPLOAD_DIR.is_dir():
        for d in config.UPLOAD_DIR.iterdir():
            if d.is_dir() and d not in referenced_dirs:
                shutil.rmtree(d, ignore_errors=True)
    # Unreferenced shared-store leftovers.
    gc_store()


# ---------------------------------------------------------------------------
# Job CRUD
# ---------------------------------------------------------------------------

def create_job(params: dict, stored_files: list,
               storage: str = "legacy") -> str:
    """Persist a new QUEUED job.

    For ``shared`` storage ``stored_files`` is a list of
    ``{"sha256", "name", "size"}`` and the same transaction both inserts the
    job and pins every referenced archive (refs += 1). If the commit fails,
    no reference is taken and the published bytes become GC-able leftovers —
    but once this returns, the job's inputs are guaranteed to stay readable.
    """
    job_id = uuid.uuid4().hex
    db = get_db()
    with _lock:
        db.execute(
            "INSERT INTO jobs (id, status, created_at, updated_at, params, "
            "files, error, progress, total, storage) "
            "VALUES (?,?,?,?,?,?,?,0,0,?)",
            (job_id, "QUEUED", _now(), _now(),
             json.dumps(params, ensure_ascii=False),
             json.dumps(stored_files, ensure_ascii=False), None, storage))
        if storage == "shared":
            for item in stored_files:
                sha = item["sha256"]
                db.execute(
                    "INSERT OR IGNORE INTO archives "
                    "(sha256, path, size, created_at, refs) "
                    "VALUES (?,?,?,?,0)",
                    (sha, str(store.store_path(sha)), item["size"], _now()))
                db.execute(
                    "UPDATE archives SET refs = refs + 1 WHERE sha256=?",
                    (sha,))
        db.commit()
    return job_id


def get_job(job_id: str) -> Optional[dict]:
    db = get_db()
    with _lock:
        row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return dict(row) if row else None


def list_jobs(limit: int = 50) -> list:
    db = get_db()
    with _lock:
        rows = db.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
            (limit,)).fetchall()
    return [dict(r) for r in rows]


def delete_job(job_id: str):
    """Delete a finished job: release its input references and remove its
    result file. Returns the deleted job dict, ``None`` if no such job, or
    the string ``"busy"`` if the job is still QUEUED/RUNNING (refused)."""
    db = get_db()
    with _lock:
        row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            return None
        job = dict(row)
        if job["status"] not in TERMINAL_STATES:
            return "busy"
        storage = job.get("storage") or "legacy"
        try:
            files = json.loads(job["files"])
        except (ValueError, TypeError):
            files = []
        db.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        if storage == "shared":
            for item in files:
                db.execute(
                    "UPDATE archives SET refs = refs - 1 WHERE sha256=?",
                    (item["sha256"],))
        db.commit()

    # Filesystem cleanup happens after the commit; anything missed by a crash
    # here is reclaimed by the next startup recovery / GC pass.
    if job["result"]:
        p = Path(job["result"])
        if p.parent == config.OUTPUT_DIR:
            try:
                p.unlink()
            except OSError:
                pass
    if storage == "legacy":
        dirs = set()
        for item in files:
            if isinstance(item, str):
                p = Path(item)
                if p.parent.parent == config.UPLOAD_DIR:
                    dirs.add(p.parent)
                    try:
                        p.unlink()
                    except OSError:
                        pass
        for d in dirs:
            try:
                d.rmdir()
            except OSError:
                pass
    else:
        # Reclaim archives whose last reference just disappeared.
        gc_store()
    return job


def _update(job_id: str, **fields) -> None:
    if not fields:
        return
    fields["updated_at"] = _now()
    sets = ", ".join(f"{k}=?" for k in fields)
    values = list(fields.values()) + [job_id]
    db = get_db()
    with _lock:
        db.execute(f"UPDATE jobs SET {sets} WHERE id=?", values)
        db.commit()


def _process_job(row: dict) -> None:
    job_id = row["id"]
    params = json.loads(row["params"])
    files = json.loads(row["files"])
    storage = row.get("storage") or "legacy"
    # Serialize all netCDF4 access process-wide; other threads may claim
    # other jobs but must wait here before touching the engine.
    with _extraction_lock:
        _run_job(job_id, params, files, storage)


def _run_job(job_id: str, params: dict, files: list, storage: str) -> None:
    try:
        sel = Selection(
            t_start=int(params["t_start"]),
            t_end=int(params["t_end"]),
            south=float(params["south"]),
            north=float(params["north"]),
            west=float(params["west"]),
            east=float(params["east"]),
        )
        if storage == "shared":
            # Resolve digests to store paths; refs >= 1 pinned at creation
            # keep them readable for the whole run.
            paths = [str(store.store_path(item["sha256"])) for item in files]
            names = [item["name"] for item in files]
            missing = [n for n, p in zip(names, paths)
                       if not os.path.isfile(p)]
            if missing:
                raise ExtractionError(
                    "共享归档缺失: " + ", ".join(missing))
        else:
            paths = files
            names = None
        output_path = str(config.OUTPUT_DIR / f"subset_{job_id}.nc")

        def progress(done: int, total: int) -> None:
            _update(job_id, progress=done, total=total)

        stats = run_extraction(paths, sel, output_path, progress,
                               display_names=names)
        # The file is fully written and renamed only at this point, so the
        # SUCCEEDED status and a complete file always appear together.
        _update(job_id, status="SUCCEEDED", progress=stats["records"],
                total=stats["records"], result=output_path,
                error=None)
    except ExtractionError as exc:
        _update(job_id, status="FAILED", error=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        _update(job_id, status="FAILED",
                error=f"内部错误: {exc}\n{traceback.format_exc()}")


def _claim_next_job() -> Optional[dict]:
    """Atomically claim the oldest QUEUED job by flipping it to RUNNING.

    The compare-and-swap (`AND status='QUEUED'`) guarantees exactly one
    worker claims a job even if several worker threads poll the same
    database — netCDF4/HDF5 extraction is never entered concurrently.
    """
    db = get_db()
    with _lock:
        row = db.execute(
            "SELECT id FROM jobs WHERE status='QUEUED' "
            "ORDER BY created_at LIMIT 1").fetchone()
        if row is None:
            return None
        cur = db.execute(
            "UPDATE jobs SET status='RUNNING', updated_at=? "
            "WHERE id=? AND status='QUEUED'",
            (_now(), row["id"]))
        if cur.rowcount != 1:
            db.rollback()
            return None  # another worker claimed it first
        db.commit()
        claimed = db.execute("SELECT * FROM jobs WHERE id=?",
                             (row["id"],)).fetchone()
    return dict(claimed) if claimed else None


def _worker_loop() -> None:
    while True:
        row = _claim_next_job()
        if row is None:
            time.sleep(1.0)
            continue
        _process_job(row)


def start_worker() -> None:
    global _worker_started
    # Guard against concurrent first requests / repeated test hook calls:
    # recovery runs once and exactly one worker thread is spawned here.
    with _start_lock:
        if _worker_started:
            return
        recover_on_startup()
        t = threading.Thread(target=_worker_loop, name="extraction-worker",
                             daemon=True)
        t.start()
        _worker_started = True


def output_path_for(job_id: str) -> Optional[Path]:
    job = get_job(job_id)
    if not job or job["status"] != "SUCCEEDED" or not job["result"]:
        return None
    p = Path(job["result"])
    # Only serve files from the managed output directory.
    if p.parent != config.OUTPUT_DIR or not p.is_file():
        return None
    return p
