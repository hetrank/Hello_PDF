from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_classic.retrievers import MultiQueryRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors.chain_extract import LLMChainExtractor

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()

#CONFIG

CHROMA_DIR = "vector_store"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

llm = ChatGroq(
  model="llama-3.3-70b-versatile",
  temperature=0
)

#Load Vector Store

def load_vector_store():
  embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

  vector_db = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
  )

  return vector_db


#Base Retriever: (MMR)

def get_base_retriever(vectordb):
  retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={
      "k":6,
      "fetch_k":20
    }
  )

  return retriever


# Multi-Query Retriever

def get_multi_query_retriver(base_retriever):

  QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
Generate 4 different versions of the given user question to improve the retrieval.

Original question : {question}

Return each version on a new line.
"""
  )

  llm_chain = QUERY_PROMPT | llm | StrOutputParser()

  return MultiQueryRetriever(
    retriever=base_retriever,
    llm_chain=llm_chain
  )


# Contextual Compression

def get_compression_retriever(multi_query_retriever):
  compressor = LLMChainExtractor.from_llm(llm)

  compression_retriever = ContextualCompressionRetriever(
    base_retriever=multi_query_retriever,
    base_compressor=compressor
  )

  return compression_retriever


# Final Retriver Pipeline:

def get_retriever():
  vectordb = load_vector_store()

  base = get_base_retriever(vectordb)
  multi = get_multi_query_retriver(base)
  final = get_compression_retriever(multi)

  return final
