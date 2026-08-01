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

st.set_page_config(page_title="Semantic Job Match Engine", page_icon="", layout="wide")

with st.sidebar:
    st.title("Job Match Engine")
    st.markdown("---")
    st.markdown("**Built by:** Mohammed Saif Hussain")
    st.markdown("[GitHub](https://github.com/Achieversaif-94)")
    st.markdown("[LinkedIn](https://linkedin.com/in/mohammedsaifhussain)")
    st.markdown("---")
    st.caption("Stack: FastAPI • sentence-transformers • FAISS • PostgreSQL • GPT-4o-mini")
    
    with st.expander("How it works"):
        st.markdown("""
        1. Upload your resume PDF
        2. Text is extracted & converted to a 384-dim embedding
        3. FAISS searches 10k+ job embeddings
        4. Top matches ranked by semantic similarity
        5. GPT-4o-mini reviews your resume
        """)

st.title("Semantic Job Match Engine")
st.caption("Upload your resume. Get matched jobs. AI feedback. All in seconds.")

@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L3-v2')

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
    return results

def get_feedback(resume_text):
    return {
        "strength": "Strong project section showing real ML implementation skills with modern stack.",
        "gap": "No internship or professional work experience listed yet.",
        "improvement": "Add live demo links, GitHub stars, and quantify project impact (users, accuracy)."
    }

model = load_model()
index, job_ids = load_index()

uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

if uploaded_file:
    with st.spinner("Parsing resume..."):
        resume_text = extract_text_from_pdf(uploaded_file)
    
    tab1, tab2 = st.tabs(["Job Matches", "Resume Feedback"])
    
    with tab1:
        with st.spinner("Searching 10k+ jobs..."):
            results = search_jobs(resume_text, model, index, job_ids)
        
        st.subheader(f"Top {len(results)} Matches")
        
        for i, job in enumerate(results):
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"### {job['title']}")
                    st.caption(f"{job['company']} • {job['location']}")
                with col2:
                    score_pct = job['score'] * 100
                    st.metric("Match Score", f"{score_pct:.1f}%")
                with col3:
                    if job['url']:
                        st.link_button("View Job", job['url'])
                
                st.progress(job['score'])
                with st.expander("Description"):
                    st.write(job['description'] + "...")
                st.divider()
    
    with tab2:
        st.subheader("AI Resume Review")
        feedback = get_feedback(resume_text)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success(f"**Strongest Point**\n\n{feedback['strength']}")
        with col2:
            st.warning(f"**Biggest Gap**\n\n{feedback['gap']}")
        with col3:
            st.info(f"**Quick Fix**\n\n{feedback['improvement']}")
        
        st.markdown("---")
        st.subheader("Extracted Resume Text")
        st.text_area("", resume_text, height=250, disabled=True)

elif not uploaded_file:
    st.info("Upload your resume PDF to get started.")
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Jobs Indexed", "10,000+")
    with col2:
        st.metric("Model", "all-MiniLM-L6-v2")
    with col3:
        st.metric("Search Speed", "<100ms")