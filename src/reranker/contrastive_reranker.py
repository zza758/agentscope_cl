from typing import List, Dict, Any


class ContrastiveReranker:
    def __init__(self, infer_engine):
        self.infer_engine = infer_engine

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 3,
        score_key: str = "contrastive_score",
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        texts = [item.get("content", "") for item in candidates]
        scores = self.infer_engine.score_pairs(query, texts)

        reranked = []
        for item, score in zip(candidates, scores):
            new_item = dict(item)
            new_item[score_key] = float(score)
            reranked.append(new_item)

        reranked.sort(key=lambda x: x[score_key], reverse=True)
        return reranked[:top_k]
