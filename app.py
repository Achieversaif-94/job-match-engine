import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Job Match Engine", page_icon="", layout="wide")

with st.sidebar:
    st.title("Job Match Engine")
    st.markdown("---")
    st.markdown("**Built by:** Mohammed Saif Hussain")
    st.markdown("[GitHub](https://github.com/Achieversaif-94)")
    st.markdown("---")
    st.caption("Stack: FastAPI • sentence-transformers • FAISS • PostgreSQL • GPT-4o-mini")

st.title("Semantic Job Match Engine")
st.caption("Upload your resume. Get matched jobs. AI feedback. All in seconds.")

uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

if uploaded_file:
    with st.spinner("Analyzing resume and searching jobs..."):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(f"{API_URL}/search", files=files)
        data = response.json()
    
    tab1, tab2 = st.tabs(["Job Matches", "Resume Feedback"])
    
    with tab1:
        st.subheader(f"Top {len(data['matches'])} Matches")
        for i, job in enumerate(data['matches']):
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
                st.progress(min(job['score'] * 3, 1.0))
                with st.expander("Description"):
                    st.write(job['description'] + "...")
                st.divider()
    
    with tab2:
        st.subheader("AI Resume Review")
        feedback_parts = data['feedback'].split('\n')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success(f"**Strongest Point**\n\n{feedback_parts[0].replace('Strongest: ', '')}")
        with col2:
            st.warning(f"**Biggest Gap**\n\n{feedback_parts[1].replace('Weakness: ', '')}")
        with col3:
            st.info(f"**Quick Fix**\n\n{feedback_parts[2].replace('Improvement: ', '')}")

elif not uploaded_file:
    st.info("Upload your resume PDF to get started.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Jobs Indexed", "10,000+")
    with col2:
        st.metric("Model", "MiniLM-L3-v2")
    with col3:
        st.metric("Search Speed", "<100ms")