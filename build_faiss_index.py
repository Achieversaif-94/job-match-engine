import os
import json
import numpy as np
import psycopg2
import faiss
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

cur.execute("SELECT id, embedding_json FROM jobs WHERE embedding_json IS NOT NULL;")
rows = cur.fetchall()

job_ids = []
embeddings = []

for job_id, emb_json in rows:
    job_ids.append(job_id)
    embeddings.append(json.loads(emb_json))

embeddings = np.array(embeddings).astype('float32')

dimension = 384
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

faiss.write_index(index, "jobs.index")

with open("job_ids.txt", "w") as f:
    for jid in job_ids:
        f.write(f"{jid}\n")

cur.close()
conn.close()

print(f"FAISS index built with {index.ntotal} vectors. Saved to jobs.index")