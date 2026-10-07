import os
import json
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("jobs.index")

with open("job_ids.txt") as f:
    job_ids = [line.strip() for line in f]

def chunk_text(text, chunk_size=400):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_resume(text):
    chunks = chunk_text(text)
    embs = model.encode(chunks, normalize_embeddings=True)
    mean = np.mean(embs, axis=0)
    norm = np.linalg.norm(mean)
    return (mean / norm).astype('float32').reshape(1, -1) if norm > 0 else None

with open("test_cases.json") as f:
    TEST_CASES = json.load(f)

results = {"recall@1": 0, "recall@3": 0, "recall@5": 0, "mrr_sum": 0.0}
n = len(TEST_CASES)

for case_name, case in TEST_CASES.items():
    query_emb = embed_resume(case["text"])
    if query_emb is None:
        continue
    scores, indices = index.search(query_emb, 5)
    retrieved = [job_ids[i] for i in indices[0]]
    relevant = set(case["relevant_job_ids"])

    hit_at = None
    if relevant & set(retrieved[:1]):
        results["recall@1"] += 1
    if relevant & set(retrieved[:3]):
        results["recall@3"] += 1
    if relevant & set(retrieved[:5]):
        results["recall@5"] += 1

    for rank, rid in enumerate(retrieved, 1):
        if rid in relevant:
            results["mrr_sum"] += 1 / rank
            hit_at = rank
            break

    print(f"{case_name}: top-5 = {retrieved}, relevant={sorted(relevant)}, first_hit_rank={hit_at}")

print()
print(f"Evaluated {n} test cases")
print(f"Recall@1: {results['recall@1']/n:.2%}")
print(f"Recall@3: {results['recall@3']/n:.2%}")
print(f"Recall@5: {results['recall@5']/n:.2%}")
print(f"MRR:      {results['mrr_sum']/n:.3f}")