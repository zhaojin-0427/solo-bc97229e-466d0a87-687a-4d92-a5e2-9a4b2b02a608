"""Content-addressed shared store for uploaded NetCDF archives.

Every uploaded file is streamed to a hidden temp file in chunks (bounded
memory), hashed with SHA-256 over its original bytes, then published into
the shared store under its digest. Publishing is atomic and no-clobber:
concurrent uploads of identical content race on ``os.link`` and exactly one
copy survives; uploads with the same file name but different content get
different digests and are stored side by side.

The store itself keeps no reference counts — that lives in the jobs table
(see ``jobs.py``). What this module *does* track is the set of in-flight
uploads (digests and temp paths currently being received / published by
this process), so the garbage collector can never reclaim a file that a
concurrent upload is about to reuse.
"""
from __future__ import annotations

import hashlib
import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from . import config

_lock = threading.Lock()
# Digests an upload in this process is using, with a reference count: two
# concurrent uploads of the same content must each hold their own guard, and
# only the last release() drops the digest from the set the GC sees.
_inflight_shas: dict = {}      # sha256 -> active upload count
_inflight_temps: set = set()   # temp paths currently being streamed to disk


def store_path(sha256: str) -> Path:
    """Location of an archive inside the shared store (sharded by prefix)."""
    return config.STORE_DIR / sha256[:2] / f"{sha256}.nc"


@dataclass
class ReceivedUpload:
    """A fully received upload: temp file on disk, digest computed."""
    sha256: str
    size: int
    tmp_path: Path


def inflight_snapshot() -> dict:
    """Consistent snapshot of in-flight digests and temp paths for the GC."""
    with _lock:
        return {
            "shas": frozenset(_inflight_shas),
            "temps": frozenset(_inflight_temps),
        }


def _track_temp(path: Path) -> None:
    with _lock:
        _inflight_temps.add(str(path))


def _untrack_temp(path: Path) -> None:
    with _lock:
        _inflight_temps.discard(str(path))


def _track_sha(sha256: str) -> None:
    with _lock:
        _inflight_shas[sha256] = _inflight_shas.get(sha256, 0) + 1


def release(sha256: str) -> None:
    """Drop one upload's guard on a digest: the upload is done and the
    archive is now either referenced by a job (safe) or unreferenced (GC may
    reclaim it once no other in-flight upload guards it)."""
    with _lock:
        n = _inflight_shas.get(sha256, 0)
        if n <= 1:
            _inflight_shas.pop(sha256, None)
        else:
            _inflight_shas[sha256] = n - 1


def receive(stream: BinaryIO, chunk_size: int = None) -> ReceivedUpload:
    """Stream an upload to a temp file while hashing it.

    Reads at most ``chunk_size`` bytes at a time, so memory stays bounded no
    matter how large the uploaded file is. The temp file is fsynced before
    this returns, and the resulting digest is immediately registered as
    in-flight so the GC cannot touch a store copy this upload will reuse.
    """
    if chunk_size is None:
        chunk_size = config.UPLOAD_CHUNK
    tmp_path = config.TMP_DIR / f".incoming-{uuid.uuid4().hex}"
    hasher = hashlib.sha256()
    size = 0
    _track_temp(tmp_path)
    try:
        with open(tmp_path, "wb") as out:
            while True:
                chunk = stream.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
                out.write(chunk)
                size += len(chunk)
            out.flush()
            os.fsync(out.fileno())
        digest = hasher.hexdigest()
        # The digest is guarded from now on; the temp stays guarded until
        # publish() (or discard()) so the GC cannot sweep it in between.
        _track_sha(digest)
        return ReceivedUpload(sha256=digest, size=size, tmp_path=tmp_path)
    except Exception:
        _untrack_temp(tmp_path)
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def publish(rec: ReceivedUpload) -> Path:
    """Move a received upload into the shared store, deduplicating by digest.

    Uses a no-clobber hard link: if a concurrent upload of the same content
    already published the digest, the existing file wins and the temp copy
    is discarded — exactly one copy remains. Returns the store path.
    """
    dest = store_path(rec.sha256)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(rec.tmp_path, dest)  # atomic, fails if dest already exists
        _fsync_dir(dest.parent)
    except FileExistsError:
        pass  # identical content already stored; the existing copy wins
    except OSError:
        # Filesystem without hard-link support: fall back to rename, but
        # never clobber a copy a concurrent upload just published.
        if not dest.exists():
            os.replace(rec.tmp_path, dest)
            _fsync_dir(dest.parent)
    try:
        rec.tmp_path.unlink()
    except OSError:
        pass
    _untrack_temp(rec.tmp_path)
    return dest


def discard(rec: ReceivedUpload) -> None:
    """Abandon a received upload that was never published (error path)."""
    release(rec.sha256)
    _untrack_temp(rec.tmp_path)
    try:
        rec.tmp_path.unlink()
    except OSError:
        pass
