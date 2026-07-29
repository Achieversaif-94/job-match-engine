import os
import fitz
from dotenv import load_dotenv

load_dotenv()

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text.strip()

def get_resume_feedback(resume_text):
    return ("Strongest: Your project section shows real implementation skills with modern ML stack.\n"
            "Weakness: No internship or work experience listed yet.\n"
            "Improvement: Add live demo link and GitHub repo with README to prove deployment skills.")

if __name__ == "__main__":
    resume_text = extract_text_from_pdf("saif_resume.pdf")
    print("Resume loaded:", len(resume_text), "chars\n")
    feedback = get_resume_feedback(resume_text)
    print("FEEDBACK:")
    print(feedback)