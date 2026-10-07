import os
import json
from dotenv import load_dotenv
import psycopg2
from sentence_transformers import SentenceTransformer

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

model = SentenceTransformer('all-MiniLM-L6-v2')

conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

cur.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS embedding_json TEXT;")
conn.commit()

cur.execute("SELECT id, description FROM jobs WHERE description IS NOT NULL;")
jobs = cur.fetchall()

for job_id, desc in jobs:
    embedding = model.encode(desc).tolist()
    cur.execute(
        "UPDATE jobs SET embedding_json = %s WHERE id = %s;",
        (json.dumps(embedding), job_id)
    )

conn.commit()
cur.close()
conn.close()

print(f"Updated {len(jobs)} jobs with embeddings.")