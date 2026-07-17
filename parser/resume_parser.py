from pypdf import PdfReader as pdf

print("reading resume in resume parser")
def resume_text_extraction(uploaded_resume):
    text = ""

    reader = pdf(uploaded_resume)

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"
    return text.strip()