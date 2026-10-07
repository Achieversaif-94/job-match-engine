# Semantic Job Match Engine

A semantic search engine that matches resumes to job listings using sentence embeddings. Deployed live on Streamlit Cloud.

**Live demo:** https://saif-job-match.streamlit.app

## What it does

Upload a PDF resume. The app extracts text, generates a sentence embedding using all-MiniLM-L6-v2, and searches a FAISS index of 30 job listings (fetched from the Adzuna API) using cosine similarity. Top-5 matches are returned with similarity scores. An LLM (Groq / gpt-oss-120b) generates three lines of resume feedback.

## Evaluation

Tested on 5 hand-labeled resume/job pairs against the 30-job index:

- Recall@5: 100%
- Recall@3: 100%
- Recall@1: 40%
- MRR: 0.667

Caveat: the index is small (30 jobs, 4-5 relevant per case), so recall@5 is easy to max out. MRR is the more informative signal - the first relevant match appears at rank 1.5 on average.

## Architecture

Resume PDF -> PyMuPDF -> text -> chunks (400 words)
all-MiniLM-L6-v2 -> 384-dim embeddings
mean-pool + normalize
FAISS cosine search against 30 job embeddings
Top-5 matches + PostgreSQL (Neon) JSON embeddings
Groq gpt-oss-120b -> 3-line feedback

## Tech stack

- Python, Streamlit for UI
- sentence-transformers (all-MiniLM-L6-v2), FAISS (IndexFlatIP on normalized vectors)
- PostgreSQL via Neon for job storage
- PyMuPDF for PDF parsing
- Adzuna API for job data
- Groq API (openai/gpt-oss-120b) for feedback

## Limitations

- 30 jobs indexed, not 10,000. The pipeline scales, but the demo dataset is small.
- Adzuna free tier truncates descriptions to about 500 chars. Full-text matching would need a paid tier.
- One embedding model, no baseline. No BM25 comparison, no reranker.
- No OCR. Scanned PDFs (image-only) return empty text.
- No no-good-match threshold. Always returns 5 results even if the resume is irrelevant.
- Similarity score is not a qualification score. It measures text similarity, not experience fit.

## Run locally

git clone https://github.com/Achieversaif-94/job-match-engine.git
cd job-match-engine
python -m venv venv
venv Scripts activate
pip install -r requirements.txt
streamlit run app.py

Create a .env file with:

ADZUNA_APP_ID=your_key
ADZUNA_APP_KEY=your_key
DATABASE_URL=your_neon_url
GROQ_API_KEY=your_groq_key

To rebuild the index and evaluate:

python generate_embeddings.py
python build_faiss_index.py
python evaluate.py

## Author

Mohammed Saif Hussain
B.Tech ECE + AI/ML, KL University
https://github.com/Achieversaif-94
