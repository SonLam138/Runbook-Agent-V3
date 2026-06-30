import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"

def call_llm(prompt):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "temperature": 0.3,
                "stream": False
            }
        )

        data = response.json()

        # ✅ debug trả về
       # print("DEBUG Ollama response:", data)

        if "response" in data:
            return data["response"]

        # nếu không có key response
        return f"LLM error: {data}"

    except Exception as e:
        return f"LLM call failed: {e}"
