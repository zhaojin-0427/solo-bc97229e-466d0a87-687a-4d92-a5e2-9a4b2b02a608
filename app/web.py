"""Flask web entry point: upload daily NetCDF archives into the shared
content-addressed store, define a subset selection, poll job status, delete
finished jobs, and download the single merged NetCDF result."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone

from flask import (Flask, abort, jsonify, redirect, render_template, request,
                   send_file, url_for)
from werkzeug.utils import secure_filename

from . import config, jobs, store

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

SAFE_NAME = re.compile(r"^[\w.\-]+$")


@app.template_filter("humanbytes")
def _humanbytes(n) -> str:
    n = float(n or 0)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024.0 or unit == "TiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.0f} B"


def _parse_epoch(value: str, field: str) -> int:
    value = (value or "").strip()
    if not value:
        raise ValueError(f"{field}不能为空")
    # Accept either a raw Unix epoch second or an ISO-8601 / datetime-local
    # string (interpreted as UTC).
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    txt = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(txt)
    except ValueError as exc:
        raise ValueError(f"{field}时间格式无效: {value}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _parse_float(value: str, field: str) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field}必须是数字") from exc


@app.before_request
def _ensure_worker() -> None:
    jobs.start_worker()


@app.route("/")
def index():
    return render_template("index.html", jobs=jobs.list_jobs(),
                           stats=jobs.store_stats())


@app.route("/jobs", methods=["POST"])
def create_job():
    try:
        t_start = _parse_epoch(request.form.get("t_start", ""), "起始时间")
        t_end = _parse_epoch(request.form.get("t_end", ""), "结束时间")
        south = _parse_float(request.form.get("south", ""), "南界纬度")
        north = _parse_float(request.form.get("north", ""), "北界纬度")
        west = _parse_float(request.form.get("west", ""), "西界经度")
        east = _parse_float(request.form.get("east", ""), "东界经度")
    except ValueError as exc:
        return render_template("error.html", message=str(exc)), 400

    uploads = request.files.getlist("files")
    if not uploads or all(not f.filename for f in uploads):
        return render_template("error.html",
                               message="请至少选择一个 NetCDF 文件"), 400

    # Receive every upload into the shared store: stream to disk in chunks
    # (bounded memory), hash the original bytes, publish by digest. Other
    # requests are not blocked — no lock is held while streaming.
    received = []   # {"sha256", "name", "size"} for the job row
    published = []  # ReceivedUpload objects already in the store
    current = None
    try:
        for seq, f in enumerate(uploads):
            original = f.filename or ""
            ext = os.path.splitext(original)[1].lower()
            if ext not in (".nc", ".netcdf", ".cdf"):
                raise ValueError(f"仅支持 NetCDF 文件: {original}")
            raw_name = secure_filename(original)
            if not raw_name or not SAFE_NAME.fullmatch(raw_name) \
                    or not raw_name.lower().endswith(
                        (".nc", ".netcdf", ".cdf")):
                # Unsafe / non-ASCII / extension-stripped name: generate a
                # safe one while preserving the real extension.
                raw_name = f"upload_{seq}{ext}"
            current = store.receive(f.stream)
            store.publish(current)
            published.append(current)
            received.append({"sha256": current.sha256, "name": raw_name,
                             "size": current.size})
            current = None
    except Exception as exc:
        if current is not None:
            store.discard(current)
        for rec in published:
            store.release(rec.sha256)
        return render_template("error.html", message=str(exc)), 400

    params = {"t_start": t_start, "t_end": t_end, "south": south,
              "north": north, "west": west, "east": east}
    try:
        # One transaction: job row + archive reference counts. If this fails,
        # the published bytes stay unreferenced and are reclaimed by the GC.
        job_id = jobs.create_job(params, received, storage="shared")
    except Exception as exc:
        for rec in published:
            store.release(rec.sha256)
        return render_template(
            "error.html", message=f"任务提交失败，请重试: {exc}"), 500
    for rec in published:
        # The job's references now protect the archives from the GC.
        store.release(rec.sha256)
    return redirect(url_for("job_detail", job_id=job_id))


def _job_inputs(job: dict) -> list:
    """Input files of a job for display: original names (+ digest prefix for
    shared-storage jobs)."""
    try:
        files = json.loads(job["files"])
    except (ValueError, TypeError):
        return []
    inputs = []
    for item in files:
        if isinstance(item, dict):
            inputs.append({"name": item.get("name", "?"),
                           "sha256": item.get("sha256", "")[:12],
                           "size": item.get("size")})
        else:
            inputs.append({"name": os.path.basename(item),
                           "sha256": None, "size": None})
    return inputs


@app.route("/jobs/<job_id>")
def job_detail(job_id: str):
    job = jobs.get_job(job_id)
    if not job:
        abort(404)
    return render_template("job.html", job=job, inputs=_job_inputs(job))


@app.route("/jobs/<job_id>/delete", methods=["POST"])
def delete_job(job_id: str):
    result = jobs.delete_job(job_id)
    if result is None:
        abort(404)
    if result == "busy":
        return render_template(
            "error.html",
            message="任务正在排队或执行中，不能删除；请等待其结束。"), 409
    return redirect(url_for("index"))


@app.route("/api/jobs/<job_id>")
def api_job(job_id: str):
    job = jobs.get_job(job_id)
    if not job:
        abort(404)
    return jsonify({
        "id": job["id"],
        "status": job["status"],
        "progress": job["progress"],
        "total": job["total"],
        "error": job["error"],
        "storage": job.get("storage") or "legacy",
        "deletable": job["status"] in jobs.TERMINAL_STATES,
        "download": (url_for("download", job_id=job["id"])
                     if job["status"] == "SUCCEEDED" else None),
    })


@app.route("/api/store")
def api_store():
    return jsonify(jobs.store_stats())


@app.route("/jobs/<job_id>/download")
def download(job_id: str):
    p = jobs.output_path_for(job_id)
    if p is None:
        abort(404)
    return send_file(str(p), as_attachment=True,
                     download_name=f"temperature_subset_{job_id}.nc",
                     mimetype="application/x-netcdf")


@app.errorhandler(413)
def too_large(_e):
    return render_template("error.html",
                           message="上传文件超过大小限制"), 413


if __name__ == "__main__":
    jobs.start_worker()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
