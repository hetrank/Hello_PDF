from pdf2image import convert_from_path
from concurrent.futures import ProcessPoolExecutor
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from paddleocr import PaddleOCR
from backend.config import POPPLER_PATH


max_workers = max(1, os.cpu_count()//2)
OCR_MODEL = None


def init_ocr():
    global OCR_MODEL
    OCR_MODEL = PaddleOCR(use_angle_cls=True, lang='en')

# PaddleOCR (worker function):

def ocr_page_paddle(args):
    img_path, page_number, source_pdf = args

    #Initialize inside process (imp for Multiprocessing)
    global OCR_MODEL

    result = OCR_MODEL.ocr(img_path, cls=True)

    text_lines = []
    if result and result[0]:
        text_lines = [line[1][0] for line in result[0]]
    
    text = "\n".join(text_lines)

    return Document(
        page_content=text,
        metadata={
            "page": page_number,
            "source": source_pdf
        }
    )


def extract_text_from_pdf_paddle(pdf_path, dpi=75, batch_size=5):
        
  documents = []
  page_start = 1

  while True:
      images = convert_from_path(
          pdf_path,
          dpi=dpi,
          first_page=page_start,
          last_page=page_start + batch_size - 1,
          output_folder="temp_images",
          paths_only=True,
          poppler_path=POPPLER_PATH
      )

      if not images:
          break
      
      args = [
          (img, page_start + i, pdf_path)
          for i, img in enumerate(images)
      ]

      with ProcessPoolExecutor(max_workers=max_workers, initializer=init_ocr) as executor:
          results = list(executor.map(ocr_page_paddle, args))
    
      documents.extend(results)
      page_start += batch_size

      if len(images) < batch_size:
          break

  for path in images:
      os.remove(path)

  return documents        

#Hybrid PDF loader (hybrid feature will be implemented later)

def pdf_to_doc(pdf_path):
    loader = PyPDFLoader(pdf_path)
    print("Loaded")
    text_documents = loader.load()
    is_scanned = True

    for pages in text_documents:
        if len(pages.page_content.strip()) > 60:
            is_scanned = False
            break
    
    if is_scanned:
        print("Scanned PDF detected → using PaddleOCR")
        scan_documents = extract_text_from_pdf_paddle(pdf_path)
        scan_documents = [doc for doc in scan_documents if len(doc.page_content.strip()) > 30]
        return scan_documents
    else:
        print("Text-based PDF detected → using PyPDFLoader")
        cleaned_docs = []
        
        for doc in text_documents:
            if len(doc.page_content.strip()) < 20:
                continue
            
            if "source" not in doc.metadata:
                doc.metadata["source"] = pdf_path

            cleaned_docs.append(doc)

        return cleaned_docs