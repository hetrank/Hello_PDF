from rag_retriever import get_retriever
from generation import generate_answer

from ragas import evaluate
from ragas.metrics import (
  faithfulness,
  answer_relevancy,
  context_precision,
  context_recall
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from datasets import Dataset
from eval_data import eval_data

llm = ChatGroq(
  model="llama-3.3-70b-versatile",
  temperature=0
)
embeddings = HuggingFaceEmbeddings(
  model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Build Dataset
def build_dataset():
  retriever = get_retriever()

  questions = []
  answers = []
  contexts = []
  ground_truths = []

  for item in eval_data:
    question = item["question"]
    gt = item["ground_truth"]

    docs = retriever.invoke(question)
    context = [doc.page_content for doc in docs]

    answer = generate_answer(retriever, question)

    questions.append(question)
    answers.append(answer)
    contexts.append(context)
    ground_truths.append(gt)
  
  dataset = Dataset.from_dict({
    "question" : questions,
    "answer" : answers,
    "contexts" : contexts,
    "ground_truth" : ground_truths
  })

  return dataset


# Run Evaluation:
def run_evaluation():
  dataset = build_dataset()

  result = evaluate(
    dataset,
    metrics=[
      faithfulness,
      answer_relevancy,
      context_precision,
      context_recall
    ],
    llm=llm,
    embeddings=embeddings
  )

  print(result)

if __name__ == "__main__":
  run_evaluation()
