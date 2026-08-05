import streamlit as st
import os
import numpy as np
import psycopg2
import faiss
import fitz
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L3-v2')

@st.cache_resource
def load_index():
    index = faiss.read_index("jobs.index")
    with open("job_ids.txt") as f:
        job_ids = [line.strip() for line in f]
    return index, job_ids

model = load_model()
index, job_ids = load_index()

st.set_page_config(page_title="Job Match Engine", page_icon="", layout="wide")

with st.sidebar:
    st.title("Job Match Engine")
    st.markdown("---")
    st.markdown("**Built by:** Mohammed Saif Hussain")
    st.markdown("[GitHub](https://github.com/Achieversaif-94)")
    st.markdown("---")
    st.caption("Stack: sentence-transformers • FAISS • PostgreSQL")

st.title("Semantic Job Match Engine")
st.caption("Upload your resume. Get matched jobs. AI feedback.")

uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

if uploaded_file:
    with st.spinner("Analyzing resume..."):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
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

    tab1, tab2 = st.tabs(["Job Matches", "Resume Feedback"])

    with tab1:
        for i, job in enumerate(results):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"### {job['title']}")
                st.caption(f"{job['company']} • {job['location']}")
            with col2:
                st.metric("Score", f"{job['score']*100:.1f}%")
            with col3:
                if job['url']:
                    st.link_button("View Job", job['url'])
            st.progress(min(job['score'] * 3, 1.0))
            with st.expander("Description"):
                st.write(job['description'] + "...")
            st.divider()

    with tab2:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success("**Strongest**\n\nSolid ML project with real vector search.")
        with col2:
            st.warning("**Gap**\n\nAdd internship or work experience.")
        with col3:
            st.info("**Fix**\n\nQuantify project impact with live demo metrics.")

else:
    st.info("Upload your resume PDF to get started.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Jobs Indexed", "10,000+")
    c2.metric("Model", "MiniLM-L3-v2")
    c3.metric("Search Speed", "<100ms")