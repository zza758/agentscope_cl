import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def format_query_text(text: str) -> str:
    text = (text or "").strip()
    return f"query: {text}"


def format_passage_text(text: str) -> str:
    text = (text or "").strip()
    return f"passage: {text}"


class ContrastiveEncoder(nn.Module):
    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name_or_path)

    def mean_pooling(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.mean_pooling(outputs.last_hidden_state, attention_mask)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings


class ContrastiveEncoderInfer:
    def __init__(self, model_dir: str, max_length: int = 128, device: str = None):
        self.model_dir = model_dir
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, fix_mistral_regex=True)
        except TypeError:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = ContrastiveEncoder(model_dir).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        emb = self.model.encode(enc["input_ids"], enc["attention_mask"])
        return emb.cpu().numpy()

    @staticmethod
    def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        query_norm = np.linalg.norm(query_vec) + 1e-12
        doc_norms = np.linalg.norm(doc_vecs, axis=1) + 1e-12
        sims = np.dot(doc_vecs, query_vec) / (doc_norms * query_norm)
        return sims

    def score_pairs(self, query: str, candidates: List[str]) -> List[float]:
        if not candidates:
            return []

        formatted_query = format_query_text(query)
        formatted_candidates = [format_passage_text(x) for x in candidates]

        query_vec = self.encode_texts([formatted_query])[0]
        cand_vecs = self.encode_texts(formatted_candidates)
        sims = self.cosine_similarity(query_vec, cand_vecs)
        return sims.tolist()
