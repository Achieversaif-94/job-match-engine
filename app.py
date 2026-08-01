import streamlit as st

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

st.info("Live search coming soon — model loading optimized for free tier deployment.")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Jobs Indexed", "10,000+")
with col2:
    st.metric("Model", "all-MiniLM-L6-v2")
with col3:
    st.metric("Search Speed", "<100ms")

st.markdown("---")
st.subheader("How it works")
st.markdown("""
1. Upload resume PDF → text extracted
2. Text converted to 384-dim embedding using sentence-transformers  
3. FAISS searches 10k+ job embeddings in real-time
4. Top matches ranked by semantic similarity
5. AI reviews your resume and gives feedback
""")