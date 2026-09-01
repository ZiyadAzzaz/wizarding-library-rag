import os

import httpx
from dotenv import load_dotenv

load_dotenv()


class RagApiClient:
    def __init__(self) -> None:
        self.base_url = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
        self.timeout = float(os.getenv("API_TIMEOUT_SECONDS", "90"))

    def health(self) -> dict:
        with httpx.Client(timeout=10) as client:
            response = client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()

    def query(self, question: str, top_k: int = 4) -> dict:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/query", json={"question": question, "top_k": top_k}
            )
            response.raise_for_status()
            return response.json()
