"""
FastAPI wrapper for HackEval Agent system.
Run with:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import uuid
import shutil
import asyncio
import traceback
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Import from your orchestrator + utils
from orchestrator import process_file, _expand_team_glob
from utils import load_eval_parameters_from_path, EvaluationParameters

# Load environment variables
load_dotenv()

# Configuration
JOB_DIR = os.getenv("JOB_DIR", "uploaded_files")
os.makedirs(JOB_DIR, exist_ok=True)

CONCURRENCY = int(os.getenv("CONCURRENCY", "2"))
AGENT_MODE = os.getenv("AGENT_MODE", "combined")
EVAL_PARAMS_FILE = os.getenv("EVAL_PARAMS_FILE", "eval_params.json")

# Semaphore for concurrent processing
semaphore = asyncio.Semaphore(CONCURRENCY)

# In-memory job store (replace with DB/Redis for production)
jobs: Dict[str, Dict[str, Any]] = {}

app = FastAPI(title="HackEval Agent API", version="1.0")

# ============================================================
# MODELS
# ============================================================

class ProcessResponse(BaseModel):
    job_id: str
    message: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    file_name: str
    started_at: str
    finished_at: Optional[str] = None
    error: Optional[str] = None


# ============================================================
# HELPERS
# ============================================================

def get_eval_params() -> EvaluationParameters:
    try:
        return load_eval_parameters_from_path(EVAL_PARAMS_FILE)
    except Exception as e:
        print(f"[WARN] Failed to load eval params: {e}")
        return EvaluationParameters.default()


async def run_job(job_id: str, file_path: str):
    """Background coroutine to run process_file() asynchronously."""
    job = jobs[job_id]
    job["status"] = "running"
    job["started_at"] = datetime.now().isoformat()

    eval_params = get_eval_params()

    try:
        ctx = await process_file(file_path, AGENT_MODE, semaphore, eval_params)
        job["status"] = "done"
        job["result"] = {
            "team_name": ctx.team_name,
            "scores": ctx.scores,
            "summary": ctx.scoring_summary,
            "feedback": ctx.feedback,
            "workflow_report": ctx.workflow_report,
            "format_notes": ctx.format_notes,
        }
    except Exception as e:
        job["status"] = "failed"
        job["error"] = f"{type(e).__name__}: {e}"
        job["traceback"] = traceback.format_exc()
    finally:
        job["finished_at"] = datetime.now().isoformat()


# ============================================================
# ROUTES
# ============================================================

@app.post("/process", response_model=ProcessResponse)
async def process_upload(file: UploadFile = File(...)):
    """
    Upload a .pdf or .pptx file to process.
    Returns a job_id you can poll with /status/{job_id} or /result/{job_id}.
    """
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {".pdf", ".ppt", ".pptx"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # Save upload
    file_path = os.path.join(JOB_DIR, f"{uuid.uuid4().hex}_{file.filename}")
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "file_name": file.filename,
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "file_path": file_path,
    }

    asyncio.create_task(run_job(job_id, file_path))
    return {"job_id": job_id, "message": f"Job {job_id} started for {file.filename}"}


@app.post("/process_glob", response_model=Dict[str, str])
async def process_glob(pattern: str = Form(...)):
    """
    Process multiple files using a glob pattern (e.g. 'data/**/*.pdf').
    """
    files = _expand_team_glob(pattern)
    if not files:
        raise HTTPException(status_code=404, detail="No matching files found")

    job_ids = {}
    for file_path in files:
        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            "file_name": os.path.basename(file_path),
            "status": "queued",
            "created_at": datetime.now().isoformat(),
            "file_path": file_path,
        }
        asyncio.create_task(run_job(job_id, file_path))
        job_ids[file_path] = job_id
    return job_ids


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    j = jobs[job_id]
    return {
        "job_id": job_id,
        "status": j["status"],
        "file_name": j["file_name"],
        "started_at": j.get("started_at"),
        "finished_at": j.get("finished_at"),
        "error": j.get("error"),
    }


@app.get("/result/{job_id}")
async def get_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    j = jobs[job_id]
    if j["status"] != "done":
        return {"status": j["status"], "message": "Job not completed yet"}
    return JSONResponse(j["result"])


@app.get("/jobs")
async def list_jobs():
    """List all jobs in memory."""
    return {
        job_id: {
            "status": j["status"],
            "file_name": j["file_name"],
            "created_at": j.get("created_at"),
            "finished_at": j.get("finished_at"),
        }
        for job_id, j in jobs.items()
    }


@app.get("/")
def root():
    return {"message": "HackEval Agent API is running!"}

@app.get("/health")
def health():
    return {"status": "ok"}
