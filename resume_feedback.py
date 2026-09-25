import os
import fitz
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text.strip()

def get_resume_feedback(resume_text):
    prompt = f"""You are a technical resume reviewer. Analyze this resume and give exactly 3 lines:
- Line 1: Strongest part of this resume
- Line 2: Biggest gap or weakness
- Line 3: One specific improvement to land a Python/ML internship

Resume:
{resume_text[:3000]}

Respond with exactly 3 lines starting with 'Strongest:', 'Weakness:', 'Improvement:'"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    resume_text = extract_text_from_pdf("saif_resume.pdf")
    print("Resume loaded:", len(resume_text), "chars\n")
    feedback = get_resume_feedback(resume_text)
    print("FEEDBACK:")
    print(feedback)