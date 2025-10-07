import fitz  # PyMuPDF
from docx import Document

def extract_text_from_pdf(pdf_path):
    """Extract and return text from a PDF file."""
    text = ""   
    try:
        with fitz.open(pdf_path) as doc: 
            for page in doc:              
                text += page.get_text()   
        return text.strip()               
    except Exception as e:
        return f"[ERROR] Could not extract PDF: {str(e)}"

if __name__ == "__main__":
    pdf_file = "Business Legal Document.pdf"

    result = extract_text_from_pdf(pdf_file)

    if result.startswith("[ERROR]"):
        print(result)
    else:
        print("[INFO] PDF text extracted successfully!\n")
        print(result) 

def extract_text_from_docx(docx_path):
    """Extract and return text from a DOCX file."""
    text = ""
    try:
        doc = Document(docx_path)
        for para in doc.paragraphs:
            if para.text.strip():  # ignore empty lines
                text += para.text.strip() + "\n"
        return text.strip()
    except Exception as e:
        return f"[ERROR] Could not extract DOCX: {str(e)}"
if __name__ == "__main__":
    docx_file = "Land_Registration_Agreement.docx"
    result = extract_text_from_docx(docx_file)
    print(result)