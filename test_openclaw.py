import requests

OPENCLAW_URL = "http://127.0.0.1:18789/api/chat"

TOKEN = " 6b620df90d6bddb319489ef2613550330b7863c50b5d1c50"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

payload = {
    "messages": [
        {
            "role": "user",
            "content": "Hello from FastAPI integration test"
        }
    ]
}

response = requests.post(
    OPENCLAW_URL,
    json=payload,
    headers=headers
)

print("STATUS:", response.status_code)
print(response.text)
