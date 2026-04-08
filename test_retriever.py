from rag_retriever import get_retriever
import time

retriever_start_time = time.time()
retriever = get_retriever()
retriever_end_time = time.time()

print(f"Got the retriever in {retriever_end_time-retriever_start_time}")

query = "What is the offender's name?"

docs = retriever.invoke(query)

for d in docs:
    print(d.metadata)
    print(d.page_content[:200])
    print("-" * 50)