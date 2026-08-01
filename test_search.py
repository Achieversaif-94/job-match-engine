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

query = "python backend developer django flask"
query_vec = model.encode(query).astype('float32').reshape(1, -1)

scores, indices = index.search(query_vec, 3)

conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
    job_id = job_ids[idx]
    cur.execute("SELECT title, company, location FROM jobs WHERE id = %s", (job_id,))
    title, company, location = cur.fetchone()
    print(f"Rank {i+1}: {title} at {company}, {location} | Score: {score:.4f}")

cur.close()
conn.close()