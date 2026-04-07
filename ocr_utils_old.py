from pdf2image import convert_from_path
from concurrent.futures import ThreadPoolExecutor
import pytesseract
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
import os
from config import TESSERACT_PATH, POPPLER_PATH


pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
max_workers = max(1, os.cpu_count()//2)


def ocr_page(img, page_number=None, source_pdf=None):
    text = pytesseract.image_to_string(img)
    metadata = {"page": page_number}
    if source_pdf:
        metadata["source"] = source_pdf
    return Document(page_content=text, metadata=metadata)


def extract_text_from_pdf(pdf_path):
        
  images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH, dpi=150, grayscale=True)

  documents = []

  with ThreadPoolExecutor(max_workers=max_workers) as executor:
      results = executor.map(lambda args: ocr_page(*args), [(img, i+1, pdf_path) for i,img in enumerate(images)])

      documents = list(results)
  
  return documents


def pdf_to_doc(pdf_path):
    loader = PyPDFLoader(pdf_path)
    print("Loaded")
    is_scanned = True
    text_documents = loader.load()

    for pages in text_documents:
        if len(pages.page_content.strip()) > 60:
            is_scanned = False
            break
    
    if is_scanned:
        return extract_text_from_pdf(pdf_path)
    else:
        print("Yes, I return from PyPDFLoader")
        for doc in text_documents:
            if "source" not in doc.metadata:
                doc.metadata["source"] = pdf_path

        return text_documents