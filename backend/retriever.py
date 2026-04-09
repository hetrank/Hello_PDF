from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_classic.retrievers import MultiQueryRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors.chain_extract import LLMChainExtractor

from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_retriever(persist_dir):
  vectordb = Chroma(
    persist_directory=persist_dir,
    embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
  )

  base_retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k":6, "fetch_k" : 20}
  )

  llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature= 0.3
  )

  multi_query = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
  )

  compressor = LLMChainExtractor.from_llm(llm)

  compression_retriever = ContextualCompressionRetriever(
    base_retriever=multi_query,
    base_compressor=compressor
  )

  return compression_retriever