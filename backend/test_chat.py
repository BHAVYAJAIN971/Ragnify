import requests, json, sys

# Use the doc we just indexed
doc_id = "9cb67dea"

print("Testing chat Q&A with streaming...\n")
print("Question: What is the name of the candidate in this resume?\n")
print("Answer: ", end="", flush=True)

resp = requests.post(
    "http://localhost:8000/chat",
    json={"doc_id": doc_id, "question": "What is the name of the candidate in this resume and what are their skills?"},
    stream=True,
    timeout=60,
)

if resp.status_code != 200:
    print("ERROR:", resp.text)
    sys.exit(1)

sources = []
answer = ""
for line in resp.iter_lines():
    if not line:
        continue
    line = line.decode("utf-8")
    if not line.startswith("data: "):
        continue
    data = json.loads(line[6:])
    if data["type"] == "sources":
        sources = data["sources"]
    elif data["type"] == "token":
        print(data["content"], end="", flush=True)
        answer += data["content"]
    elif data["type"] == "done":
        break
    elif data["type"] == "error":
        print("\nERROR:", data["content"])
        sys.exit(1)

print("\n")
print("Sources cited:", sources[:3])
print("\nSUCCESS! Full pipeline working with Gemini!")
