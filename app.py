from fastapi import FastAPI, UploadFile, File
from utils.extractor import extract_skills_from_resume
import shutil
from pathlib import Path

app = FastAPI(title="Resume Skill Extractor")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/extract-skills/")
async def extract_resume_skills(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    skills = extract_skills_from_resume(file_path)
    return {"filename": file.filename, "skills": skills}

@app.get("/")
async def home():
    return {"message": "Resume Skill Extractor API is running ✅"}
