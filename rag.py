import PyPDF2

def get_rag_content(pdf_path):
    text = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text.strip())

    return "\n\n".join(text)
