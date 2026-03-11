import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import jieba
from agentscope.message import TextBlock
from agentscope.tool import ToolResponse


def normalize_text(text: str) -> str:
    """
    归一化文本：
    - 转小写
    - 去除首尾空白
    - 去除中间多余空白
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", "", text)
    return text


def tokenize_zh(text: str) -> List[str]:
    """
    中文分词
    """
    text = normalize_text(text)
    return [w.strip() for w in jieba.lcut(text) if w.strip()]


def char_ngrams(text: str, n: int = 2) -> List[str]:
    """
    字符级 n-gram，用于增强中文短文本匹配鲁棒性
    """
    text = normalize_text(text)
    if len(text) < n:
        return [text] if text else []
    return [text[i:i + n] for i in range(len(text) - n + 1)]


class SimpleKnowledgeBase:
    def __init__(
        self,
        kb_path: str,
        default_top_k: int = 3,
        score_threshold: Optional[float] = None,
    ):
        self.kb_path = Path(kb_path)
        self.default_top_k = default_top_k
        self.default_score_threshold = score_threshold
        self.docs = self._load_docs()

        # 运行时注入
        self.mysql_logger = None
        self.current_task_run_id = None

        # 检索日志开关（用于消融实验）
        self.enable_retrieval_logging = True

        print(f"[KB] 已加载文档数: {len(self.docs)}")

    def bind_logger(self, mysql_logger) -> None:
        """
        绑定数据库日志器
        """
        self.mysql_logger = mysql_logger

    def set_runtime_context(self, task_run_id: Optional[int]) -> None:
        """
        在每个任务运行前设置当前 task_run_id，
        以便工具被 agent 调用时能把检索结果写入 retrieval_logs。
        """
        self.current_task_run_id = task_run_id

    def set_logging_enabled(self, enabled: bool) -> None:
        """
        控制是否写 retrieval_logs，便于消融实验。
        """
        self.enable_retrieval_logging = enabled

    def _load_docs(self) -> List[Dict]:
        docs = []
        if not self.kb_path.exists():
            print(f"[KB] 知识库文件不存在: {self.kb_path}")
            return docs

        with open(self.kb_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))
        return docs

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        基于混合打分的简单检索：
        1. phrase_score：query 归一化后整体是否包含于文档
        2. token_overlap：分词交集
        3. bigram_overlap：字符二元组交集
        """
        query_norm = normalize_text(query)
        query_terms = set(tokenize_zh(query))
        query_bigrams = set(char_ngrams(query, n=2))

        scored = []

        for doc in self.docs:
            text = doc.get("content", "")
            text_norm = normalize_text(text)
            text_terms = set(tokenize_zh(text))
            text_bigrams = set(char_ngrams(text, n=2))

            phrase_score = 0.0
            if query_norm in text_norm:
                phrase_score += 5.0

            token_overlap = len(query_terms & text_terms)
            bigram_overlap = len(query_bigrams & text_bigrams)

            score = phrase_score + 2.0 * token_overlap + 0.5 * bigram_overlap

            if score > 0:
                scored.append(
                    {
                        "doc_id": doc.get("doc_id", "unknown"),
                        "chunk_id": doc.get("chunk_id"),
                        "content": text,
                        "score": float(score),
                    }
                )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _log_retrieval_results(self, query: str, results: List[Dict]) -> None:
        """
        将检索结果写入 retrieval_logs。
        受 enable_retrieval_logging 开关控制。
        """
        if not self.enable_retrieval_logging:
            return

        if self.mysql_logger is None or self.current_task_run_id is None:
            return

        for idx, item in enumerate(results, start=1):
            self.mysql_logger.log_retrieval(
                task_run_id=self.current_task_run_id,
                query_text=query,
                doc_id=item["doc_id"],
                chunk_id=item.get("chunk_id"),
                rank_pos=idx,
                score=item.get("score"),
                used_flag=1,
            )

    async def retrieve_knowledge(
        self,
        query: str,
        limit: int = 3,
        score_threshold: Optional[float] = None,
    ) -> ToolResponse:
        """
        AgentScope 工具函数：
        根据 query 检索知识库，返回 ToolResponse。

        Args:
            query: 检索问题
            limit: 返回条数
            score_threshold: 最低分数阈值

        Returns:
            ToolResponse
        """
        if limit is None:
            limit = self.default_top_k

        if score_threshold is None:
            score_threshold = self.default_score_threshold

        results = self.search(query=query, top_k=limit)

        if score_threshold is not None:
            results = [r for r in results if r["score"] >= score_threshold]

        self._log_retrieval_results(query=query, results=results)

        if not results:
            text = "未检索到相关知识。"
        else:
            formatted = []
            for i, item in enumerate(results, start=1):
                formatted.append(
                    f"[知识{i}] doc_id={item['doc_id']} "
                    f"chunk_id={item['chunk_id']} "
                    f"score={item['score']}\n{item['content']}"
                )
            text = "\n\n".join(formatted)

        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=text,
                )
            ]
        )
