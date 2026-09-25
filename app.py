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
    return SentenceTransformer('paraphrase-MiniLM-L3-v2')

@st.cache_resource(show_spinner="Loading FAISS index...")
def load_index():
    index = faiss.read_index("jobs.index")
    with open("job_ids.txt") as f:
        job_ids = [line.strip() for line in f]
    return index, job_ids

def get_resume_feedback(resume_text):
    prompt = f"""You are a technical resume reviewer. Give EXACTLY 3 lines, nothing else:

Strongest: <one sentence>
Weakness: <one sentence>
Improvement: <one sentence>

Resume:
{resume_text[:3000]}"""
    try:
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=250
        )
        return response.choices[0].message.content
    except Exception:
        return "Strongest: Strong project portfolio.\nWeakness: Unable to generate feedback right now.\nImprovement: Try again in a moment."

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

            resume_vec = model.encode(resume_text).astype('float32').reshape(1, -1)
            scores, indices = index.search(resume_vec, 5)

            conn = psycopg2.connect(DATABASE_URL)
            cur = conn.cursor()
            results = []
            for score, idx in zip(scores[0], indices[0]):
                job_id = job_ids[idx]
                cur.execute("SELECT title, company, location, description, redirect_url FROM jobs WHERE id = %s", (job_id,))
                title, company, location, desc, url = cur.fetchone()
                display_score = max(0.0, min(float(score), 1.0))
                results.append({
                    "title": title,
                    "company": company,
                    "location": location,
                    "score": display_score,
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
                    st.metric("Score", f"{job['score']*100:.1f}%")
                with col3:
                    if job['url']:
                        st.link_button("View Job", job['url'])
                st.progress(job['score'])
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
    c2.metric("Model", "MiniLM-L3-v2")
    c3.metric("Search Speed", "<100ms")