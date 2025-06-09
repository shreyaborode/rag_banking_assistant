import requests

API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/paraphrase-MiniLM-L6-v2"
API_TOKEN = "your_huggingface_api_token"
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

payload = {"inputs": "Test query"}
response = requests.post(API_URL, headers=HEADERS, json=payload)

if response.status_code == 200:
    print("Model is working!")
else:
    print(f"Error: {response.status_code} - {response.text}")