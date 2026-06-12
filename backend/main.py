from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import ensure_data_directories, settings
from .jobs import create_job, get_job, submit_processing_job, update_job
from .processor import process_video
from .utils import safe_filename


ensure_data_directories()

app = FastAPI(
    title="Intelligent Football Highlight Detection API",
    description=(
        "MVP API for detecting probable football highlight candidates "
        "using lightweight audio analysis."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check() -> dict:
    return {
        "status": "ok",
        "project": "Intelligent Football Highlight Detection System",
        "mvp": "audio_only_v1",
    }


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)) -> dict:
    filename = safe_filename(file.filename or "uploaded_video.mp4")
    extension = Path(filename).suffix.lower()

    if extension not in settings.ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension: {extension}",
        )

    job_id = create_job(input_filename=filename, input_path="")

    saved_path = settings.UPLOAD_DIR / f"{job_id}_{filename}"

    try:
        with saved_path.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                output.write(chunk)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not save upload: {exc}")

    update_job(job_id, input_path=str(saved_path))
    submit_processing_job(job_id, process_video, saved_path)

    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Video uploaded. Processing has started in the background.",
    }


@app.get("/status/{job_id}")
def get_status(job_id: str) -> dict:
    job = get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "stage": job.get("stage"),
        "progress": job.get("progress", 0),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "error": job.get("error"),
    }


@app.get("/results/{job_id}")
def get_results(job_id: str) -> dict:
    job = get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    if job["status"] != "completed":
        return {
            "job_id": job_id,
            "status": job["status"],
            "stage": job.get("stage"),
            "progress": job.get("progress", 0),
            "error": job.get("error"),
            "message": "Results are available only after processing is completed.",
        }

    return job["result"]


@app.get("/download/{job_id}")
def download_highlights(job_id: str) -> FileResponse:
    job = get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail="The job is not completed yet.",
        )

    final_video_path = job.get("result", {}).get("final_video_path")

    if not final_video_path:
        raise HTTPException(
            status_code=404,
            detail="No final highlights video was generated.",
        )

    path = Path(final_video_path)
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Final highlights video file is missing on disk.",
        )

    return FileResponse(
        path=str(path),
        media_type="video/mp4",
        filename=f"highlights_{job_id}.mp4",
    )


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
