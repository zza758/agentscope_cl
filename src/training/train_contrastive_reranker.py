import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# 把 src 目录加入 Python 搜索路径
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW

from src.utils.config_loader import load_config, PROJECT_ROOT


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    if not path.exists():
        return records

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class ContrastiveMemoryDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        return {
            "query": item["query"],
            "positive": item["positive_memory"],
            "negative": item["negative_memory"],
        }


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

    def forward(
        self,
        q_input_ids,
        q_attention_mask,
        p_input_ids,
        p_attention_mask,
        n_input_ids,
        n_attention_mask,
    ):
        q_emb = self.encode(q_input_ids, q_attention_mask)
        p_emb = self.encode(p_input_ids, p_attention_mask)
        n_emb = self.encode(n_input_ids, n_attention_mask)
        return q_emb, p_emb, n_emb


def collate_fn_builder(tokenizer, max_length: int):
    def collate_fn(batch):
        queries = [x["query"] for x in batch]
        positives = [x["positive"] for x in batch]
        negatives = [x["negative"] for x in batch]

        q_enc = tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        p_enc = tokenizer(
            positives,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        n_enc = tokenizer(
            negatives,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return {
            "q_input_ids": q_enc["input_ids"],
            "q_attention_mask": q_enc["attention_mask"],
            "p_input_ids": p_enc["input_ids"],
            "p_attention_mask": p_enc["attention_mask"],
            "n_input_ids": n_enc["input_ids"],
            "n_attention_mask": n_enc["attention_mask"],
        }

    return collate_fn


def train():
    config = load_config()
    contrastive_cfg = config["contrastive"]

    if not contrastive_cfg.get("train_enabled", False):
        print("contrastive.train_enabled = false，跳过训练。")
        return

    train_data_path = Path(contrastive_cfg["train_data_path"])
    if not train_data_path.is_absolute():
        train_data_path = PROJECT_ROOT / train_data_path

    output_dir = Path(contrastive_cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(train_data_path)
    if not records:
        print(f"未找到训练数据或数据为空: {train_data_path}")
        return

    print(f"读取训练样本数: {len(records)}")

    tokenizer = AutoTokenizer.from_pretrained(contrastive_cfg["model_name_or_path"])
    model = ContrastiveEncoder(contrastive_cfg["model_name_or_path"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = ContrastiveMemoryDataset(records)
    dataloader = DataLoader(
        dataset,
        batch_size=contrastive_cfg.get("batch_size", 8),
        shuffle=True,
        collate_fn=collate_fn_builder(
            tokenizer,
            max_length=contrastive_cfg.get("max_length", 128),
        ),
    )

    optimizer = AdamW(
        model.parameters(),
        lr=float(contrastive_cfg.get("learning_rate", 2e-5)),
    )

    criterion = nn.TripletMarginLoss(
        margin=float(contrastive_cfg.get("margin", 0.2)),
        p=2,
    )

    num_epochs = int(contrastive_cfg.get("num_epochs", 3))

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0

        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            q_emb, p_emb, n_emb = model(
                batch["q_input_ids"],
                batch["q_attention_mask"],
                batch["p_input_ids"],
                batch["p_attention_mask"],
                batch["n_input_ids"],
                batch["n_attention_mask"],
            )

            loss = criterion(q_emb, p_emb, n_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(dataloader), 1)
        print(f"Epoch {epoch + 1}/{num_epochs} - avg_loss: {avg_loss:.6f}")

    save_dir = output_dir / "contrastive_encoder"
    save_dir.mkdir(parents=True, exist_ok=True)

    model.encoder.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print(f"模型已保存到: {save_dir}")


if __name__ == "__main__":
    train()
