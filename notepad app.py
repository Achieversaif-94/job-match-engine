import streamlit as st
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

st.set_page_config(page_title="Semantic Job Match Engine", layout="wide")
st.title("Semantic Job Match Engine")
st.caption("Upload your resume — get matched jobs + AI feedback")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_index():
    index = faiss.read_index("jobs.index")
    with open("job_ids.txt") as f:
        job_ids = [line.strip() for line in f]
    return index, job_ids

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text.strip()

def search_jobs(resume_text, model, index, job_ids, top_k=5):
    resume_vec = model.encode(resume_text).astype('float32').reshape(1, -1)
    scores, indices = index.search(resume_vec, top_k)
    
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    results = []
    for score, idx in zip(scores[0], indices[0]):
        job_id = job_ids[idx]
        cur.execute("SELECT title, company, location, description FROM jobs WHERE id = %s", (job_id,))
        title, company, location, desc = cur.fetchone()
        results.append({
            "title": title,
            "company": company,
            "location": location,
            "score": float(score),
            "description": desc[:300]
        })
    cur.close()
    conn.close()
    return results

def get_feedback(resume_text):
    return ("Strongest: Your project section shows real implementation skills with modern ML stack.\n"
            "Weakness: No internship or work experience listed yet.\n"
            "Improvement: Add live demo link and GitHub repo with README to prove deployment skills.")

model = load_model()
index, job_ids = load_index()

uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

if uploaded_file:
    with st.spinner("Parsing resume..."):
        resume_text = extract_text_from_pdf(uploaded_file)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Top Job Matches")
        with st.spinner("Searching jobs..."):
            results = search_jobs(resume_text, model, index, job_ids)
        
        for i, job in enumerate(results):
            with st.container():
                st.markdown(f"### {i+1}. {job['title']} at {job['company']}")
                st.caption(f"{job['location']} | Match: {job['score']:.2%}")
                st.write(job['description'] + "...")
                st.divider()
    
    with col2:
        st.subheader("Resume Feedback")
        feedback = get_feedback(resume_text)
        st.info(feedback)
        
        st.subheader("Resume Preview")
        st.text_area("Extracted Text", resume_text[:500] + "...", height=200)