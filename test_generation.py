from rag_retriever import get_retriever
from generation import generate_answer
import time

start = time.time()

start_retr = time.time()
retriever = get_retriever()
end_retr = time.time()
print(f"Got the retriever in {end_retr-start_retr}")

query = "Is Kishorbhai an offender?"

start_ans = time.time()
answer = generate_answer(retriever, query)
end_ans = time.time()
print(f"Got the ans in {end_ans-start_ans}")

print(answer)

end = time.time()
print(f"Time to test entire generation : {end-start}")