from typing import List, Dict, Any


class ContrastiveReranker:
    def __init__(self, infer_engine):
        self.infer_engine = infer_engine

    def rerank(self, query: str, candidates, top_k=None):
        if not candidates:
            return []

        texts = [x.get("content", "") for x in candidates]
        scores = self.infer_engine.score_pairs(query=query, candidates=texts)

        enriched = []
        for item, score in zip(candidates, scores):
            x = dict(item)
            x["contrastive_score"] = float(score)
            enriched.append(x)

        enriched.sort(key=lambda x: x["contrastive_score"], reverse=True)

        if top_k is None:
            return enriched
        return enriched[:top_k]
