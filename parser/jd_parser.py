from pypdf import PdfReader as pdf

print("reading jd in jd parser")
def jd_text_extraction(uploaded_jd):
    text = ""

    reader = pdf(uploaded_jd)

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"
    return text.strip()