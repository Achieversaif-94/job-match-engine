import os
import json
import numpy as np
import psycopg2
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
index = faiss.read_index("jobs.index")

with open("job_ids.txt") as f:
    job_ids = [line.strip() for line in f]

with open("dummy_resume.txt", encoding="utf-8") as f:
    resume_text = f.read()

resume_vec = model.encode(resume_text).astype('float32').reshape(1, -1)

scores, indices = index.search(resume_vec, 5)

conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

print("Top 5 Job Matches for your resume:\n")
for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
    job_id = job_ids[idx]
    cur.execute("SELECT title, company, location, description FROM jobs WHERE id = %s", (job_id,))
    title, company, location, desc = cur.fetchone()
    print(f"Rank {i+1}: {title} at {company}, {location}")
    print(f"Score: {score:.4f}")
    print(f"Description snippet: {desc[:200]}...")
    print("---")

cur.close()
conn.close()