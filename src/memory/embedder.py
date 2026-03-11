from typing import List
from openai import OpenAI
import numpy as np


class DashScopeEmbedder:
    def __init__(self, api_key: str, model_name: str = "text-embedding-v4", normalize: bool = True):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model_name = model_name
        self.normalize = normalize

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        resp = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
        )

        vectors = [item.embedding for item in resp.data]

        if self.normalize:
            normalized = []
            for vec in vectors:
                arr = np.array(vec, dtype=np.float32)
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
                normalized.append(arr.tolist())
            return normalized

        return vectors

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]
