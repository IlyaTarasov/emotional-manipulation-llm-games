import json
import time
import urllib.request

try:
    import requests
except Exception:
    requests = None


class OllamaModel:
    def __init__(self, model_name="llama3", temperature=0.1, num_predict=220):
        self.model_name = model_name
        self.temperature = temperature
        self.num_predict = num_predict
        self.url = "http://localhost:11434/api/generate"
        self.session = None
        if requests is not None:
            # Disable proxy for localhost
            self.session = requests.Session()
            self.session.trust_env = False  # Ignore proxy environment variables

    def generate(self, prompt, system_prompt=""):
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict
            }
        }

        last_error = None
        for attempt in range(1, 4):
            try:
                if self.session is not None:
                    # Use session with disabled proxy
                    r = self.session.post(self.url, json=payload, timeout=180)
                    data = r.json()
                else:
                    request = urllib.request.Request(
                        self.url,
                        data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(request, timeout=180) as response:
                        data = json.loads(response.read().decode("utf-8"))
                break
            except Exception as e:
                last_error = e
                if attempt < 3:
                    time.sleep(3 * attempt)
        else:
            return f"ERROR_REQUEST: {last_error}"

        if "response" in data:
            return data["response"]

        if "message" in data and "content" in data["message"]:
            return data["message"]["content"]

        if "error" in data:
            return f"ERROR_MODEL: {data['error']}"

        return json.dumps(data, ensure_ascii=False)
