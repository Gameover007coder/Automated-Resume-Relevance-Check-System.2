# parser.py
import pypdf
import docx2txt
import io

def extract_text_from_pdf(file):
    """Extracts text from a PDF file stream."""
    pdf_reader = pypdf.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    """Extracts text from a DOCX file stream."""
    return docx2txt.process(file)

def parse_resume(file):
    """Parses a resume file (PDF or DOCX) and returns its text content."""
    filename = file.name
    content_stream = io.BytesIO(file.getvalue())
    
    if filename.endswith(".pdf"):
        return extract_text_from_pdf(content_stream)
    elif filename.endswith(".docx"):
        return extract_text_from_docx(content_stream)
    else:
        raise ValueError("Unsupported file format. Please upload a .pdf or .docx file.")