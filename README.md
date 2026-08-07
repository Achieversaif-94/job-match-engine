\# Semantic Job Match Engine



\[!\[Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://saif-job-match.streamlit.app)

\[!\[Python](https://img.shields.io/badge/python-3.14-blue)](https://python.org)

\[!\[Streamlit](https://img.shields.io/badge/deployed-Streamlit\_Cloud-red)](https://streamlit.io)



Semantic job matching engine using sentence-transformers, FAISS vector search, and GPT-4o-mini for resume feedback.



\## Architecture



Resume PDF > PyMuPDF > Text > MiniLM-L3 Embedding > FAISS Search > Top-5 Matches



Job Data (Adzuna) > PostgreSQL (Neon) > FAISS Index > GPT-4o-mini Feedback



\## Features



\- Upload resume PDF, automatic text extraction

\- Semantic search across 30+ live job listings

\- FAISS vector similarity, sub-100ms search

\- Match scores with confidence percentages

\- AI-powered resume feedback

\- Live deployment



\## Tech Stack



Python | FastAPI | Streamlit | sentence-transformers | FAISS | PostgreSQL | PyMuPDF | Adzuna API | GPT-4o-mini



\## Live Demo



https://saif-job-match.streamlit.app



\## Run Locally



