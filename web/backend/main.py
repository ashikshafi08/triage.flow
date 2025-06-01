from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from processing import process_issue
from store import jobs
import uuid
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend's URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class IssueRequest(BaseModel):
    issue_url: str
    prompt_type: str  # "explain", "fix", "test", or "summarize"

@app.post("/process_issue")
async def handle_issue(request: IssueRequest, background_tasks: BackgroundTasks):
    # Validate prompt type
    valid_types = ["explain", "fix", "test", "summarize", "document", "review", "prioritize"]
    if request.prompt_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid prompt type. Must be one of: {', '.join(valid_types)}"
        )
    
    # Create job ID
    job_id = f"job_{uuid.uuid4().hex}"
    
    # Store initial job status
    jobs[job_id] = {"status": "processing", "result": None, "error": None, "progress_log": []}
    
    # Add background task
    background_tasks.add_task(process_issue, job_id, request.issue_url, request.prompt_type)
    
    return {"job_id": job_id, "status": "processing"}

@app.get("/job_status/{job_id}")
async def get_job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

# @app.get("/api/files")
# async def list_files():
#     file_list = []
#     for root, dirs, files in os.walk("."):
#         # Skip hidden files and directories
#         dirs[:] = [d for d in dirs if not d.startswith('.')]
#         for f in files:
#             if not f.startswith('.'):
#                 rel_path = os.path.relpath(os.path.join(root, f), ".")
#                 file_list.append({"path": rel_path})
#     return file_list
