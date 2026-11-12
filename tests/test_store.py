"""Tests for the shared content-addressed archive store: deduplication,
reference counting, job deletion, garbage collection and crash recovery.

Run from the repository root:
    python -I -m unittest discover -s tests -v
"""
import atexit
import hashlib
import io
import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path

# Share the isolated DATA_HOME with the other test modules in this process.
_TMP_HOME = Path(os.environ.setdefault(
    "NC_TEST_HOME", tempfile.mkdtemp(prefix="nc-home-store-")))
os.environ["DATA_HOME"] = str(_TMP_HOME)
os.environ["TIME_CHUNK"] = "2"
atexit.register(shutil.rmtree, str(_TMP_HOME), ignore_errors=True)

from app import config, jobs, store
from app.web import app
from scripts.make_samples import T0, DAY, make_samples

PARAMS = {"t_start": T0, "t_end": T0 + 3 * DAY,
          "south": -10, "north": 10, "west": 170.0, "east": -170.0}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reset_store():
    """Empty the shared store and both DB tables (jobs are terminal here)."""
    db = jobs.get_db()
    with jobs._lock:
        db.execute("DELETE FROM jobs")
        db.execute("DELETE FROM archives")
        db.commit()
    shutil.rmtree(config.STORE_DIR, ignore_errors=True)
    config.STORE_DIR.mkdir(parents=True, exist_ok=True)
    for p in config.TMP_DIR.glob(".incoming-*"):
        p.unlink()


class StoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="nc-storetests-"))
        cls.info = make_samples(cls.tmp / "samples")
        cls.file_bytes = [Path(f).read_bytes() for f in cls.info["files"]]
        jobs.start_worker()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        _reset_store()

    # -- helpers -----------------------------------------------------------

    def _shared_job(self, blobs, names=None, params=None):
        """Receive+publish blobs and create a shared-storage job."""
        names = names or [f"f{i}.nc" for i in range(len(blobs))]
        items, recs = [], []
        for data, name in zip(blobs, names):
            rec = store.receive(io.BytesIO(data))
            store.publish(rec)
            recs.append(rec)
            items.append({"sha256": rec.sha256, "name": name,
                          "size": rec.size})
        jid = jobs.create_job(params or PARAMS, items, storage="shared")
        for rec in recs:
            store.release(rec.sha256)
        return jid

    def _manual_job(self, status, blobs, names=None):
        """Insert a shared-storage job row directly in a non-QUEUED state
        (the worker never touches it), with references pinned exactly as
        create_job would."""
        import uuid as _uuid
        names = names or [f"f{i}.nc" for i in range(len(blobs))]
        items, recs = [], []
        for data, name in zip(blobs, names):
            rec = store.receive(io.BytesIO(data))
            store.publish(rec)
            recs.append(rec)
            items.append({"sha256": rec.sha256, "name": name,
                          "size": rec.size})
        jid = _uuid.uuid4().hex
        db = jobs.get_db()
        with jobs._lock:
            db.execute(
                "INSERT INTO jobs (id, status, created_at, updated_at, "
                "params, files, error, progress, total, storage) "
                "VALUES (?,?,?,?,?,?,?,0,0,?)",
                (jid, status, jobs._now(), jobs._now(),
                 json.dumps(PARAMS), json.dumps(items), None, "shared"))
            for item in items:
                db.execute(
                    "INSERT OR IGNORE INTO archives "
                    "(sha256, path, size, created_at, refs) VALUES (?,?,?,?,0)",
                    (item["sha256"], str(store.store_path(item["sha256"])),
                     item["size"], jobs._now()))
                db.execute(
                    "UPDATE archives SET refs = refs + 1 WHERE sha256=?",
                    (item["sha256"],))
            db.commit()
        for rec in recs:
            store.release(rec.sha256)
        return jid

    def _wait(self, job_id, timeout=30):
        deadline = time.time() + timeout
        while time.time() < deadline:
            j = jobs.get_job(job_id)
            if j["status"] in ("SUCCEEDED", "FAILED"):
                return j
            time.sleep(0.1)
        self.fail("job did not finish in time")

    def _store_files(self):
        return sorted(config.STORE_DIR.rglob("*.nc"))

    # -- tests -------------------------------------------------------------

    def test_01_dedup_same_content_one_copy(self):
        """Two jobs uploading identical bytes share a single stored file;
        stats report one archive and the real byte count."""
        blob = self.file_bytes[0]
        j1 = self._shared_job([blob], ["day1.nc"])
        j2 = self._shared_job([blob, blob], ["a.nc", "b.nc"])
        files = self._store_files()
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].name, _sha(blob) + ".nc")
        stats = jobs.store_stats()
        self.assertEqual(stats["count"], 1)
        self.assertEqual(stats["bytes"], len(blob))
        # refs: 1 (job1) + 2 (job2 references it twice)
        with jobs._lock:
            refs = jobs.get_db().execute(
                "SELECT refs FROM archives WHERE sha256=?",
                (_sha(blob),)).fetchone()["refs"]
        self.assertEqual(refs, 3)
        # Both jobs can run independently off the shared copy.
        for jid in (j1, j2):
            self.assertEqual(self._wait(jid)["status"], "SUCCEEDED")

    def test_02_same_name_different_content_kept_separately(self):
        j1 = self._shared_job([self.file_bytes[0]], ["day.nc"])
        j2 = self._shared_job([self.file_bytes[2]], ["day.nc"])
        self.assertEqual(len(self._store_files()), 2)
        stats = jobs.store_stats()
        self.assertEqual(stats["count"], 2)
        self.assertEqual(stats["bytes"],
                         len(self.file_bytes[0]) + len(self.file_bytes[2]))
        for jid in (j1, j2):
            self.assertEqual(self._wait(jid)["status"], "SUCCEEDED")

    def test_03_concurrent_uploads_single_copy(self):
        """Many threads receiving+publishing identical content concurrently
        leave exactly one copy in the store."""
        blob = self.file_bytes[1]
        barrier = threading.Barrier(8)
        errors = []

        def uploader():
            try:
                rec = store.receive(io.BytesIO(blob))
                barrier.wait(timeout=10)
                store.publish(rec)
                store.release(rec.sha256)
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=uploader) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        self.assertEqual(errors, [])
        files = self._store_files()
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].stat().st_size, len(blob))

    def test_04_gc_skips_inflight_upload(self):
        """The GC must not reclaim a store file that a concurrent upload is
        reusing, even while its archive row is still unreferenced."""
        blob = self.file_bytes[0]
        rec = store.receive(io.BytesIO(blob))
        dest = store.publish(rec)          # digest now in-flight protected
        # Simulate the window before the job commit: an unreferenced row.
        db = jobs.get_db()
        with jobs._lock:
            db.execute(
                "INSERT INTO archives (sha256, path, size, created_at, refs)"
                " VALUES (?,?,?,?,0)",
                (rec.sha256, str(dest), rec.size, jobs._now()))
            db.commit()
        jobs.gc_store()
        self.assertTrue(dest.is_file())          # not reclaimed: in-flight
        self.assertEqual(jobs.store_stats()["count"], 1)
        store.release(rec.sha256)                # upload abandoned
        jobs.gc_store()
        self.assertFalse(dest.exists())          # now reclaimable
        self.assertEqual(jobs.store_stats()["count"], 0)

    def test_05_delete_finished_job_releases_inputs_and_result(self):
        jid = self._shared_job(self.file_bytes[:2], ["d1.nc", "d2.nc"])
        job = self._wait(jid)
        self.assertEqual(job["status"], "SUCCEEDED")
        result = Path(job["result"])
        self.assertTrue(result.is_file())
        self.assertEqual(jobs.store_stats()["count"], 2)

        deleted = jobs.delete_job(jid)
        self.assertEqual(deleted["id"], jid)
        self.assertIsNone(jobs.get_job(jid))
        self.assertFalse(result.exists())            # result removed
        self.assertEqual(self._store_files(), [])    # archives reclaimed
        self.assertEqual(jobs.store_stats()["count"], 0)
        self.assertIsNone(jobs.delete_job(jid))      # already gone

    def test_06_delete_refused_while_active(self):
        # A RUNNING job (as the worker would see it mid-extraction) cannot be
        # deleted; its archives stay pinned.
        jid = self._manual_job("RUNNING", self.file_bytes[:1])
        self.assertEqual(jobs.delete_job(jid), "busy")
        self.assertEqual(len(self._store_files()), 1)  # archive kept
        self.assertEqual(jobs.store_stats()["count"], 1)
        # Web layer maps the refusal to 409.
        client = app.test_client()
        resp = client.post(f"/jobs/{jid}/delete")
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(client.post("/jobs/nope/delete").status_code, 404)

    def test_07_archive_pinned_while_any_job_references_it(self):
        blob = self.file_bytes[0]
        j1 = self._shared_job([blob])
        j2 = self._shared_job([blob])
        for jid in (j1, j2):
            self.assertEqual(self._wait(jid)["status"], "SUCCEEDED")
        jobs.delete_job(j1)
        # One reference remains: the archive must survive GC.
        jobs.gc_store()
        self.assertEqual(len(self._store_files()), 1)
        self.assertEqual(jobs.store_stats()["count"], 1)
        jobs.delete_job(j2)
        self.assertEqual(self._store_files(), [])

    def test_08_restart_recovery_keeps_accepted_inputs_reclaims_rest(self):
        """After a crash+restart, an accepted job's inputs stay complete and
        readable; unreferenced leftovers are reclaimed."""
        # A job that was RUNNING when the process died (worker never picks
        # up RUNNING rows, so this is deterministic).
        jid = self._manual_job("RUNNING", self.file_bytes[:2],
                               ["d1.nc", "d2.nc"])
        # Leftover 1: published upload whose job commit never happened.
        orphan = store.receive(io.BytesIO(self.file_bytes[2]))
        orphan_dest = store.publish(orphan)
        store.release(orphan.sha256)
        # Leftover 2: archive row with refs=0 (failed commit after insert).
        stray = b"stray bytes"
        stray_sha = _sha(stray)
        stray_path = store.store_path(stray_sha)
        stray_path.parent.mkdir(parents=True, exist_ok=True)
        stray_path.write_bytes(stray)
        with jobs._lock:
            jobs.get_db().execute(
                "INSERT INTO archives (sha256, path, size, created_at, refs)"
                " VALUES (?,?,?,?,0)",
                (stray_sha, str(stray_path), len(stray), jobs._now()))
            jobs.get_db().commit()
        # Leftover 3: temp file of an interrupted upload.
        stale_tmp = config.TMP_DIR / ".incoming-deadbeef"
        stale_tmp.write_bytes(b"partial")
        self.assertEqual(jobs.store_stats()["count"], 3)

        # Simulate process restart.
        jobs._worker_started = False
        jobs.recover_on_startup()
        jobs._worker_started = True

        job = jobs.get_job(jid)
        # The in-flight job is failed by recovery (restart semantics) ...
        self.assertEqual(job["status"], "FAILED")
        # ... but its accepted inputs remain complete and readable.
        items = json.loads(job["files"])
        for item, blob in zip(items, self.file_bytes[:2]):
            p = store.store_path(item["sha256"])
            self.assertTrue(p.is_file())
            self.assertEqual(_sha(p.read_bytes()), _sha(blob))
        # Unreferenced leftovers are all reclaimed.
        self.assertFalse(orphan_dest.exists())
        self.assertFalse(stray_path.exists())
        self.assertFalse(stale_tmp.exists())
        self.assertEqual(jobs.store_stats()["count"], 2)

        # The inputs are genuinely usable: extraction straight from the
        # store succeeds after the restart.
        from app.engine import Selection, run_extraction
        out = str(self.tmp / "post_restart.nc")
        stats = run_extraction(
            [str(store.store_path(i["sha256"])) for i in items],
            Selection(T0, T0 + 3 * DAY, -10, 10, 170.0, -170.0), out)
        self.assertEqual(stats["records"], 4)
        self.assertTrue(os.path.isfile(out))

    def test_09_web_flow_upload_dedup_download_delete(self):
        client = app.test_client()
        from werkzeug.datastructures import MultiDict

        def submit(blobs, names):
            form = MultiDict()
            for data, name in zip(blobs, names):
                form.add("files", (io.BytesIO(data), name))
            for key, val in [("t_start", str(PARAMS["t_start"])),
                             ("t_end", str(PARAMS["t_end"])),
                             ("south", "-10"), ("north", "10"),
                             ("west", "170"), ("east", "-170")]:
                form.add(key, val)
            resp = client.post("/jobs", data=form,
                               content_type="multipart/form-data")
            self.assertEqual(resp.status_code, 302)
            return resp.headers["Location"].rstrip("/").split("/")[-1]

        # Same file name, different content -> stored separately.
        j1 = submit([self.file_bytes[0], self.file_bytes[1]],
                    ["day.nc", "day.nc"])
        self.assertEqual(jobs.store_stats()["count"], 2)
        # Re-upload of the first file -> deduplicated, no new bytes.
        j2 = submit([self.file_bytes[0]], ["copy.nc"])
        stats = jobs.store_stats()
        self.assertEqual(stats["count"], 2)
        self.assertEqual(stats["bytes"],
                         len(self.file_bytes[0]) + len(self.file_bytes[1]))

        for jid in (j1, j2):
            self.assertEqual(self._wait(jid)["status"], "SUCCEEDED")

        # Index page shows the shared-archive count and real bytes.
        html = client.get("/").get_data(as_text=True)
        self.assertIn("共享归档", html)
        self.assertIn("<b>2</b> 个", html)

        # Delete the first job through the web: its unique archive is
        # reclaimed, the shared one stays; result download is gone.
        resp = client.post(f"/jobs/{j1}/delete")
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(client.get(f"/jobs/{j1}/download").status_code, 404)
        self.assertEqual(jobs.store_stats()["count"], 1)
        # The surviving archive still serves the second job's download.
        self.assertEqual(client.get(f"/jobs/{j2}/download").status_code, 200)
        client.post(f"/jobs/{j2}/delete")
        self.assertEqual(jobs.store_stats()["count"], 0)
        self.assertEqual(self._store_files(), [])

    def test_10_legacy_jobs_keep_working(self):
        """Jobs stored the pre-shared-store way still extract, download and
        delete; their input files outside the managed upload dir survive."""
        jid = jobs.create_job(PARAMS, self.info["files"])  # legacy default
        job = self._wait(jid)
        self.assertEqual(job["status"], "SUCCEEDED")
        self.assertEqual(job["storage"], "legacy")
        client = app.test_client()
        self.assertEqual(client.get(f"/jobs/{jid}/download").status_code, 200)
        result = Path(job["result"])
        deleted = jobs.delete_job(jid)
        self.assertEqual(deleted["storage"], "legacy")
        self.assertFalse(result.exists())
        for f in self.info["files"]:
            self.assertTrue(Path(f).is_file())  # samples untouched

    def test_11_receive_reads_in_bounded_chunks(self):
        """receive() never pulls more than UPLOAD_CHUNK bytes per read, so
        memory stays flat regardless of upload size."""
        blob = self.file_bytes[0] * 3

        class SpyStream:
            def __init__(self, data):
                self._buf = io.BytesIO(data)
                self.max_read = 0
                self.reads = 0

            def read(self, n=-1):
                self.max_read = max(self.max_read, n)
                self.reads += 1
                return self._buf.read(n)

        old_chunk = config.UPLOAD_CHUNK
        config.UPLOAD_CHUNK = 4096  # force many chunks for a small blob
        try:
            spy = SpyStream(blob)
            rec = store.receive(spy)
        finally:
            config.UPLOAD_CHUNK = old_chunk
        self.assertEqual(rec.sha256, _sha(blob))
        self.assertEqual(rec.size, len(blob))
        self.assertGreater(spy.reads, 2)  # actually streamed in chunks
        self.assertGreater(spy.max_read, 0)
        self.assertLessEqual(spy.max_read, 4096)
        store.discard(rec)
        self.assertFalse(rec.tmp_path.exists())

    def test_12_failed_job_delete_still_releases_references(self):
        # East boundary -160 includes the planted conflict -> job fails.
        params = dict(PARAMS, east=-160.0)
        jid = self._shared_job(self.file_bytes, params=params)
        job = self._wait(jid)
        self.assertEqual(job["status"], "FAILED")
        self.assertIn("冲突", job["error"])
        # Error message uses the original upload names, not store digests.
        self.assertIn("f1.nc", job["error"])
        self.assertEqual(jobs.store_stats()["count"], 3)
        jobs.delete_job(jid)
        self.assertEqual(jobs.store_stats()["count"], 0)
        self.assertEqual(self._store_files(), [])

    def test_13_gc_skips_digest_guarded_by_second_upload(self):
        """Two concurrent uploads of identical content each hold a guard on
        the digest; the first upload's release (and even its job's deletion)
        must not expose the file while the second upload is still reusing
        it."""
        blob = self.file_bytes[0]
        sha = _sha(blob)
        rec_a = store.receive(io.BytesIO(blob))
        rec_b = store.receive(io.BytesIO(blob))   # digest guarded twice
        dest = store.publish(rec_a)
        store.publish(rec_b)                      # reuses A's copy
        jid = jobs.create_job(
            PARAMS, [{"sha256": sha, "name": "a.nc", "size": len(blob)}],
            storage="shared")
        store.release(rec_a.sha256)               # A done; B still guarded
        self.assertEqual(self._wait(jid)["status"], "SUCCEEDED")
        jobs.delete_job(jid)                      # refs -> 0, GC runs here
        self.assertTrue(dest.is_file())           # B's guard kept it alive
        self.assertEqual(jobs.store_stats()["count"], 1)
        # B can still commit its job against the surviving copy.
        jid2 = jobs.create_job(
            PARAMS, [{"sha256": sha, "name": "b.nc", "size": len(blob)}],
            storage="shared")
        store.release(rec_b.sha256)
        self.assertEqual(self._wait(jid2)["status"], "SUCCEEDED")


if __name__ == "__main__":
    unittest.main()
