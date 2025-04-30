import json
import os

import requests


# Get API key from environment
api_key = os.environ.get("OPENROUTER_API_KEY")

# Headers
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

# Simple test of the API
url = "https://openrouter.ai/api/v1/chat/completions"
data = {
    "model": "meta-llama/llama-3.3-70b-instruct",
    "messages": [{"role": "user", "content": "Say hello"}],
}

try:
    response = requests.post(url, headers=headers, json=data)
    print(f"Status code: {response.status_code}")
    print(f"Raw response: {response.text}")

    # Try to parse as JSON
    try:
        json_response = response.json()
        print(f"JSON parsed successfully: {json.dumps(json_response, indent=2)}")
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")

except Exception as e:
    print(f"Request error: {e}")
