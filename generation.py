from langchain_groq import  ChatGroq
from langchain_core.prompts import PromptTemplate


# LLM:
llm = ChatGroq(
  model="llama-3.3-70b-versatile",
  temperature=0
)


# Format Documents:
def format_docs(docs):
  formatted = []

  for doc in docs:
    source = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page", "unknown")

    chunk = f"[Source: {source}, Page {page}]\n{doc.page_content}"
    formatted.append(chunk)
  
  return "\n\n".join(formatted)


# Prompt
PROMPT = PromptTemplate(
  input_variables=["context", "question"],
  template="""
You are a helpful research assistant.

Use ONLY the provided context to answer the question.

For every statement you make, include a citation in this format:
[Source: filename, Page X]

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


# Generate Answer
def generate_answer(retriever, query):
  docs = retriever.invoke(query)

  if not docs:
    return "No relevant documents found."
  
  context = format_docs(docs)

  prompt = PROMPT.format(
    context = context,
    question = query
  )

  response = llm.invoke(prompt)

  return response.content