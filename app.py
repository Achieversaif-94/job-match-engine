import streamlit as st
import os
import numpy as np
import psycopg2
import faiss
import fitz
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

@st.cache_resource(show_spinner="Loading embedding model...")
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource(show_spinner="Loading FAISS index...")
def load_index():
    index = faiss.read_index("jobs.index")
    with open("job_ids.txt") as f:
        job_ids = [line.strip() for line in f]
    return index, job_ids

def chunk_text(text, chunk_size=150):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_resume(model, resume_text):
    chunks = chunk_text(resume_text)
    if not chunks:
        return None
    chunk_embs = model.encode(chunks, normalize_embeddings=True)
    mean_emb = np.mean(chunk_embs, axis=0)
    norm = np.linalg.norm(mean_emb)
    if norm > 0:
        mean_emb = mean_emb / norm
    return mean_emb.astype('float32').reshape(1, -1)

def get_resume_feedback(resume_text):
    prompt = f"""Give exactly 3 short lines about this resume, one sentence each:

Strongest: 
Weakness: 
Improvement: 

Resume: {resume_text[:2000]}"""
    try:
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_completion_tokens=1024,
            reasoning_effort="low"
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            return "Strongest: Strong project portfolio.\nWeakness: Unable to generate feedback right now.\nImprovement: Try again in a moment."
        return content
    except Exception as e:
        return f"Strongest: Strong project portfolio.\nWeakness: {str(e)[:80]}\nImprovement: Try again in a moment."

st.set_page_config(page_title="Job Match Engine", page_icon="", layout="wide")

with st.sidebar:
    st.title("Job Match Engine")
    st.markdown("---")
    st.markdown("**Built by:** Mohammed Saif Hussain")
    st.markdown("[GitHub](https://github.com/Achieversaif-94)")
    st.markdown("---")
    st.caption("Stack: sentence-transformers • FAISS • PostgreSQL • Groq")

st.title("Semantic Job Match Engine")
st.caption("Upload your resume. Get matched jobs. AI feedback.")

uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")
st.caption("Resumes are processed in-memory and sent to Groq's API for feedback. Not stored on our servers.")

if uploaded_file:
    try:
        model = load_model()
        index, job_ids = load_index()

        with st.spinner("Analyzing resume..."):
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            resume_text = ""
            for page in doc:
                resume_text += page.get_text()
            doc.close()

            if not resume_text.strip():
                st.error("Could not extract text from this PDF. Please upload a digital PDF, not a scanned image.")
                st.stop()

            resume_vec = embed_resume(model, resume_text)
            if resume_vec is None:
                st.error("Could not generate embedding from resume text.")
                st.stop()

            scores, indices = index.search(resume_vec, 5)

            top_score = float(scores[0][0])
            if top_score < 0.30:
                st.warning("No strong matches found. Your resume doesn't closely match any indexed jobs. Try a different resume or wait for more jobs to be indexed.")

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

        with st.spinner("Generating AI feedback..."):
            feedback = get_resume_feedback(resume_text)

        tab1, tab2 = st.tabs(["Job Matches", "Resume Feedback"])

        with tab1:
            for i, job in enumerate(results):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"### {job['title']}")
                    st.caption(f"{job['company']} • {job['location']}")
                with col2:
                    st.metric("Similarity", f"{job['score']*100:.1f}%")
                with col3:
                    if job['url']:
                        st.link_button("View Job", job['url'])
                st.progress(min(max(job['score'], 0.0), 1.0))
                with st.expander("Description"):
                    st.write(job['description'] + "...")
                st.divider()

        with tab2:
            st.subheader("AI Resume Review")
            lines = [l.strip() for l in feedback.split('\n') if l.strip()]
            labels = ["Strongest", "Weakness", "Improvement"]
            icons = [st.success, st.warning, st.info]
            cols = st.columns(3)
            for i in range(3):
                with cols[i]:
                    if i < len(lines):
                        text = lines[i].split(':', 1)[1].strip() if ':' in lines[i] else lines[i]
                    else:
                        text = "Not available"
                    icons[i](f"**{labels[i]}**\n\n{text}")

    except Exception as e:
        st.error(f"Something went wrong: {str(e)}")
        st.info("Try re-uploading your resume, or check that it's a valid PDF.")

else:
    st.info("Upload your resume PDF to get started.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Jobs Indexed", "30")
    c2.metric("Model", "MiniLM-L6")
    c3.metric("Search Speed", "<100ms")