import fitz
import sys

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text.strip()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_resume.py <path_to_resume.pdf>")
    else:
        text = extract_text_from_pdf(sys.argv[1])
        print(text[:1000])
        print(f"\n--- Total chars: {len(text)} ---")