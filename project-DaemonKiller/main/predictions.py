import json
from rag import RAG
from generation import generate

rag=RAG(n_retrieved=10)

jsonpath = '/teamspace/studios/this_studio/project-DaemonKiller/data/evaluation_prompts.json'

with open(jsonpath, encoding='utf-8') as f:
    data = json.load(f)
print(data)
# responses = []
for item in data:
    query = item["input"]
    context = rag.query(query)
    response = generate(query=query, context=context)
    item["prediction"] = response
    print(response)
with open("updated_predictions.json", "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

