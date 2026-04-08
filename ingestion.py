from ocr_utils_old import pdf_to_doc
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import shutil
import os

# CONFIGURATION
text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=800,
  chunk_overlap=100
)

CHROMA_DIR = "vector_store"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# Load Documents
def load_documents(pdf_paths):
  all_docs = []

  for path in pdf_paths:
    print(f"Processing: {path}")
    docs = pdf_to_doc(path)

    docs = [d for d in docs if len(d.page_content.strip()) > 30]

    all_docs.extend(docs)
  
  return all_docs


# Split it into chunks
def split_documents(documents):
  chunks = text_splitter.split_documents(documents)

  for chunk in chunks:
    if "source" not in chunk.metadata:
      chunk.metadata["source"] = "unknown"
    if "page" not in chunk.metadata:
      chunk.metadata["page"] = -1
  
  return chunks


# Create Vector Store
def create_vector_store(chunks):
  #deleting the existing DB(OVERWRITE MODE)
  if os.path.exists(CHROMA_DIR):
    shutil.rmtree(CHROMA_DIR)

  embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
  )

  vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory = CHROMA_DIR
  )

  vectordb.persist()

  return vectordb


# Main Ingestion Pipeline
def ingest_pdfs(pdf_paths):
  documents = load_documents(pdf_paths)
  print(f"Loaded {len(documents)} pages")

  chunks = split_documents(documents)
  print(f"Created {len(chunks)} chunks")

  vectordb = create_vector_store(chunks)
  print("Vector store created successfully")

  return vectordb


#Test Run
if __name__ == "__main__":
  pdf_paths = [
    r"C:\MY STUFF\DAIICT\GenAI\Research-Assistant\sample_pdfs\Challan - 2 Reciept.pdf",
  ]

  ingest_pdfs(pdf_paths)