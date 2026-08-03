from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import numpy as np
import psycopg2
import faiss
import fitz
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
index = faiss.read_index("jobs.index")

with open("job_ids.txt") as f:
    job_ids = [line.strip() for line in f]

@app.get("/")
def root():
    return {"status": "Job Match Engine API", "version": "1.0"}

@app.post("/search")
async def search_jobs(file: UploadFile = File(...)):
    contents = await file.read()
    doc = fitz.open(stream=contents, filetype="pdf")
    resume_text = ""
    for page in doc:
        resume_text += page.get_text()
    doc.close()

    resume_vec = model.encode(resume_text).astype('float32').reshape(1, -1)
    scores, indices = index.search(resume_vec, 5)

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    results = []
    for score, idx in zip(scores[0], indices[0]):
        job_id = job_ids[idx]
        cur.execute("SELECT title, company, location, description, redirect_url FROM jobs WHERE id = %s", (job_id,))
        title, company, location, desc, url = cur.fetchone()
        results.append({
            "title": title,
            "company": company,
            "location": location,
            "score": float(score),
            "description": desc[:300],
            "url": url
        })
    cur.close()
    conn.close()

    feedback = "Strongest: Solid ML project implementation.\nWeakness: Add internship experience.\nImprovement: Quantify project impact with metrics."

    return {"matches": results, "feedback": feedback}