from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from retriever import get_retriever
import os
import json
from dotenv import load_dotenv

def generate_answer(chat_id, question):
  persist_dir = os.path.join("vector_store", f"chat_{chat_id}")

  retriever = get_retriever(persist_dir)

  docs = retriever.invoke(question)

  if not docs:
    return "No relevant documents found.", []

  context = "\n\n".join([
    f"[Source: {doc.metadata.get('source')}, Page {doc.metadata.get('page')}]\\n{doc.page_content}"
    for doc in docs
  ])

  prompt =HumanMessage(content=f"""
You are a helpful research assistant.

Use ONLY the provided context to answer the question.

Explain the topic asked about in the question if needed and dont just copy paste the provided context. 

If the answer is not in the context, say:
"I could not find the answer in the provided documents."

Be concise. Prefer bullet points where possible.

--------------------------------------
context:
{context}
--------------------------------------

Question:
{question}

Answer:
"""
)

  llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.4,
    streaming=True
  )

  for chunk in llm.stream([prompt]):
    if chunk.content:
      yield chunk.content + "\n"

  sources = list(set([
    f"{doc.metadata.get('source')} (Page {doc.metadata.get('page')})"
    for doc in docs
  ]))

  yield "\n\n__SOURCES__" + json.dumps(sources)