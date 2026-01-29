import requests

queries = ["test_query"]

payload = {
    "queries": queries,
    "topk": 3,
    "return_scores": True
}

res = requests.post("http://placeholder-retriever:8000/retrieve", json=payload)

print(res.content)