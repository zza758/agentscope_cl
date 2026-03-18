import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from agentscope.message import Msg

from src.memory.memory_record import MemoryRecord
from src.memory.quality_gate import assess_memory_quality
from src.runtime.task_context import TaskContext
from src.utils.structured_answer import parse_structured_answer


def extract_text_from_response(response) -> str:
    """从 AgentScope 返回对象中尽量稳定地提取纯文本答案。"""
    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif hasattr(block, "type") and getattr(block, "type", None) == "text":
                    texts.append(getattr(block, "text", ""))
            if texts:
                return "\n".join(texts)
        elif isinstance(content, str):
            return content
    return str(response)


class TaskRunner:
    def __init__(
        self,
        agent,
        memory_manager,
        mysql_logger,
        knowledge_base,
        ablation_cfg,
        memory_top_k: int = 3,
        contrastive_cfg: dict = None,
        contrastive_reranker=None,
        memory_policy=None,
        experiment_id: str = "default_exp",
        enable_profile: bool = False,
        enable_kb_runtime_context: bool = True,
        quality_gate_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.agent = agent
        self.memory_manager = memory_manager
        self.mysql_logger = mysql_logger
        self.knowledge_base = knowledge_base
        self.ablation_cfg = ablation_cfg
        self.memory_top_k = memory_top_k
        self.contrastive_cfg = contrastive_cfg or {}
        self.contrastive_reranker = contrastive_reranker
        self.memory_policy = memory_policy
        self.experiment_id = experiment_id
        self.enable_profile = enable_profile
        self.enable_kb_runtime_context = enable_kb_runtime_context
        self.quality_gate_cfg = quality_gate_cfg or {}

    def _retrieve_memories(
        self,
        query: str,
        task_context: TaskContext,
        task_type: Optional[str] = None,
        task_entity: Optional[str] = None,
        final_top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if final_top_k is None:
            final_top_k = self.memory_top_k
        if not self.ablation_cfg.get("use_memory", True):
            return []
        if self.memory_manager is None:
            return []

        candidate_top_k = self.memory_top_k
        if self.contrastive_cfg.get("rerank_enabled", False):
            candidate_top_k = int(self.contrastive_cfg.get("candidate_top_k", self.memory_top_k))

        memory_items = self.memory_manager.retrieve_memory_with_scores(
            query=query,
            task_context=task_context,
            top_k=candidate_top_k,
            task_type=task_type,
            task_entity=task_entity,
        )
        if (
            self.ablation_cfg.get("use_contrastive_rerank", False)
            and self.contrastive_cfg.get("rerank_enabled", False)
            and self.contrastive_reranker is not None
            and memory_items
        ):
            rerank_top_k = None if self.memory_policy is not None else int(
                self.contrastive_cfg.get("final_top_k", self.memory_top_k)
            )
            memory_items = self.contrastive_reranker.rerank(
                query=query,
                candidates=memory_items,
                top_k=rerank_top_k,
            )

        if self.memory_policy is not None and memory_items:
            memory_items = self.memory_policy.select_memories(
                query=query,
                task_context=task_context,
                candidates=memory_items,
            )
        else:
            memory_items = memory_items[:final_top_k]
        return memory_items

    @staticmethod
    def _format_memory_items(memory_items: List[Dict[str, Any]]) -> str:
        if not memory_items:
            return "暂无历史经验。"
        lines = []
        for idx, item in enumerate(memory_items, start=1):
            base_score = item.get("score")
            rerank_score = item.get("contrastive_score")
            policy_score = item.get("policy_score")
            tag = "支持经验" if item.get("is_support_memory") else "历史经验"
            parts = [f"[{tag}{idx}"]
            if base_score is not None:
                parts.append(f"base={base_score:.4f}")
            if rerank_score is not None:
                parts.append(f"rerank={rerank_score:.4f}")
            if policy_score is not None:
                parts.append(f"policy={policy_score:.4f}")
            prefix = "|".join(parts) + "]"

            meta = []
            if item.get("task_type"):
                meta.append(f"task_type={item['task_type']}")
            if item.get("entity"):
                meta.append(f"entity={item['entity']}")
            if item.get("memory_quality"):
                meta.append(f"quality={item['memory_quality']}")
            meta_text = f" ({'; '.join(meta)})" if meta else ""
            lines.append(f"{prefix}{meta_text} {item.get('content', '')}")
        return "\n".join(lines)

    def _get_support_memories(
        self,
        support_task_ids: List[str],
        task_context: TaskContext,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not support_task_ids:
            return []
        if self.memory_manager is None:
            return []
        if not hasattr(self.memory_manager, "get_memories_by_task_ids"):
            return []
        return self.memory_manager.get_memories_by_task_ids(
            task_ids=support_task_ids,
            task_context=task_context,
            limit=limit,
        )

    @staticmethod
    def _merge_memory_items(
        support_items: List[Dict[str, Any]],
        retrieved_items: List[Dict[str, Any]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """
        合并 support-aware memory 与普通 retrieval memory：
        1. support memory 永远优先
        2. 按 task_id + 规范化文本去重
        3. 最终截断到 limit
        """
        merged: List[Dict[str, Any]] = []
        seen_task_ids = set()
        seen_texts = set()

        def _norm_text(x: str) -> str:
            return " ".join((x or "").strip().split()).lower()

        for item in support_items + retrieved_items:
            task_id = item.get("task_id")
            text = item.get("memory_summary") or item.get("content", "")
            norm_text = _norm_text(text)
            if task_id and task_id in seen_task_ids:
                continue
            if norm_text and norm_text in seen_texts:
                continue

            merged.append(item)
            if task_id:
                seen_task_ids.add(task_id)
            if norm_text:
                seen_texts.add(norm_text)
            if len(merged) >= limit:
                break
        return merged

    def _maybe_log_profile(
        self,
        task_run_id: int,
        task_id: str,
        task_order: int,
        stream_id: Optional[str],
        profile: Dict[str, Any],
    ) -> None:
        if not self.enable_profile:
            return
        if hasattr(self.mysql_logger, "log_profile"):
            self.mysql_logger.log_profile(
                task_run_id=task_run_id,
                task_id=task_id,
                task_order=task_order,
                stream_id=stream_id,
                profile=profile,
            )

    async def run_single_task(
        self,
        task: Optional[Dict[str, Any]] = None,
        *,
        task_id: Optional[str] = None,
        task_order: Optional[int] = None,
        query: Optional[str] = None,
        stream_id: Optional[str] = None,
        task_type: Optional[str] = None,
        entity: Optional[str] = None,
        support_task_ids: Optional[List[str]] = None,
        source_dataset: Optional[str] = None,
        source_sample_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if task is not None:
            task_id = task["task_id"]
            task_order = int(task["task_order"])
            query = task["query"]
            stream_id = task.get("stream_id")
            task_type = task.get("task_type")
            entity = task.get("entity")
            support_task_ids = task.get("support_task_ids") or []
            source_dataset = task.get("source_dataset")
            source_sample_id = task.get("source_sample_id")
            meta = task.get("meta") or {}

        support_task_ids = support_task_ids or []
        meta = meta or {}

        wall_start = time.perf_counter()
        task_start_time = datetime.now()
        task_run_id = self.mysql_logger.log_task_run(
            experiment_id=self.experiment_id,
            task_id=task_id,
            task_order=task_order,
            query_text=query,
            task_start_time=task_start_time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        task_context = TaskContext(
            experiment_id=self.experiment_id,
            task_id=task_id,
            task_order=task_order,
            task_run_id=task_run_id,
            task_start_time=task_start_time,
            stream_id=stream_id,
            task_type=task_type,
            task_entity=entity,
            support_task_ids=support_task_ids,
            source_dataset=source_dataset,
            source_sample_id=source_sample_id,
            meta=meta,
        )

        if (
            self.knowledge_base is not None
            and self.ablation_cfg.get("use_knowledge_base", True)
            and self.enable_kb_runtime_context
        ):
            self.knowledge_base.set_runtime_context(task_run_id)
        elif self.knowledge_base is not None:
            self.knowledge_base.set_runtime_context(None)

        t_retrieve_start = time.perf_counter()
        if task_type == "decomposition_qa":
            effective_top_k = min(self.memory_top_k, 2)
        else:
            effective_top_k = max(self.memory_top_k, 4)

        support_memory_items = self._get_support_memories(
            support_task_ids=support_task_ids,
            task_context=task_context,
            limit=effective_top_k,
        )
        raw_support_memory_items = support_memory_items
        support_memory_items = self._filter_support_memories(support_memory_items)
        retrieved_memory_items = self._retrieve_memories(
            query=query,
            task_context=task_context,
            task_type=task_type,
            task_entity=entity,
            final_top_k=effective_top_k,
        )
        memory_items = self._merge_memory_items(
            support_items=support_memory_items,
            retrieved_items=retrieved_memory_items,
            limit=effective_top_k,
        )

        if raw_support_memory_items:
            self.mysql_logger.log_memory(
                task_run_id=task_run_id,
                memory_key=f"task:{task_id}:support_retrieve_raw",
                operation_type="support_retrieve_raw",
                memory_content=self._format_memory_items(raw_support_memory_items),
                relevance_score=raw_support_memory_items[0].get("score"),
            )
        if support_memory_items:
            self.mysql_logger.log_memory(
                task_run_id=task_run_id,
                memory_key=f"task:{task_id}:support_retrieve",
                operation_type="support_retrieve",
                memory_content=self._format_memory_items(support_memory_items),
                relevance_score=support_memory_items[0].get("score"),
            )

        t_retrieve_end = time.perf_counter()
        memories = [item.get("content", "") for item in memory_items]
        formatted_memories = self._format_memory_items(memory_items)
        if self.ablation_cfg.get("use_memory_logging", True):
            self.mysql_logger.log_memory(
                task_run_id=task_run_id,
                memory_key=f"task:{task_id}:retrieve",
                operation_type="retrieve",
                memory_content=formatted_memories,
                relevance_score=memory_items[0]["score"] if memory_items else None,
            )

        metadata_lines = []
        if stream_id:
            metadata_lines.append(f"- stream_id: {stream_id}")
        if task_type:
            metadata_lines.append(f"- task_type: {task_type}")
        if entity:
            metadata_lines.append(f"- entity: {entity}")
        if support_task_ids:
            metadata_lines.append(f"- support_task_ids: {support_task_ids}")
        if source_dataset:
            metadata_lines.append(f"- source_dataset: {source_dataset}")
        task_meta_block = "\n".join(metadata_lines) if metadata_lines else "- 无"

        if task_type == "decomposition_qa":
            task_instruction = (
                "这是一个分解子问题。优先直接给出简洁、明确的事实答案；"
                "仅在确有必要时调用知识检索工具。"
            )
        else:
            task_instruction = "必要时调用知识检索工具，并结合历史经验完成当前任务。"

        user_prompt = (
            f"当前任务:\n{query}\n\n"
            f"任务元数据:\n{task_meta_block}\n\n"
            f"历史经验:\n{formatted_memories}\n\n"
            f"{task_instruction}\n\n"
            f"请严格按以下格式输出：\n"
            f"〖最终答案〗不超过3句。\n"
            f"〖记忆摘要〗不超过80字，只保留以后可复用的结论。\n"
            f"〖策略备注〗1句话。\n"
        )
        t_compose_end = time.perf_counter()

        self.mysql_logger.log_trajectory(
            task_run_id=task_run_id,
            step_no=1,
            agent_name="MainAgent",
            action_type="compose_input",
            action_input={
                "query": query,
                "stream_id": stream_id,
                "task_type": task_type,
                "entity": entity,
                "support_task_ids": support_task_ids,
                "memories": memories,
            },
            action_output=user_prompt,
        )

        msg = Msg(name="user", content=user_prompt, role="user")
        t_agent_start = time.perf_counter()
        response = await self.agent(msg)
        t_agent_end = time.perf_counter()

        raw_answer = extract_text_from_response(response)
        parsed = parse_structured_answer(raw_answer)
        final_answer = parsed["final_answer"].strip()
        memory_summary = parsed["memory_summary"].strip()
        strategy_note = parsed["strategy_note"].strip()

        if not final_answer:
            final_answer = raw_answer.strip()
        if not memory_summary:
            memory_summary = final_answer
        if not strategy_note:
            strategy_note = "可作为后续相似任务的参考经验。"

        memory_record = MemoryRecord(
            experiment_id=self.experiment_id,
            task_id=task_id,
            task_order=task_order,
            query=query,
            answer_raw=final_answer,
            memory_summary=memory_summary,
            strategy_note=strategy_note,
            created_at=MemoryRecord.now_ts(),
            stream_id=stream_id,
            task_type=task_type,
            entity=entity,
            support_task_ids=support_task_ids,
            source_dataset=source_dataset,
            source_sample_id=source_sample_id,
            meta=meta,
        )

        gate_result = assess_memory_quality(
            query=query,
            final_answer=final_answer,
            memory_summary=memory_summary,
            task_type=task_type,
            cfg=self.quality_gate_cfg,
        )
        memory_record.memory_quality = gate_result["memory_quality"]
        memory_record.contains_placeholder = gate_result["contains_placeholder"]
        memory_record.contains_unknown = gate_result["contains_unknown"]
        memory_record.gate_passed = gate_result["gate_passed"]
        memory_record.gate_reason = gate_result["gate_reason"]

        self.mysql_logger.log_memory(
            task_run_id=task_run_id,
            memory_key=f"task:{task_id}:write_candidate",
            operation_type="write_candidate",
            memory_content=json.dumps(memory_record.to_log_dict(), ensure_ascii=False),
            relevance_score=None,
        )

        policy_allow = True
        if self.memory_policy is not None:
            policy_allow = self.memory_policy.should_write_memory(
                query=query,
                task_context=task_context,
                final_answer=final_answer,
                memory_summary=memory_summary,
                strategy_note=strategy_note,
            )
        quality_allow = memory_record.gate_passed
        should_write = policy_allow and quality_allow

        t_memory_write_start = time.perf_counter()
        memory_written = False
        if (
            should_write
            and self.ablation_cfg.get("use_memory", True)
            and self.ablation_cfg.get("use_memory_write", True)
            and self.memory_manager is not None
        ):
            self.memory_manager.write_memory(memory_record)
            memory_written = True
            self.mysql_logger.log_memory(
                task_run_id=task_run_id,
                memory_key=f"task:{task_id}:write",
                operation_type="write",
                memory_content=json.dumps(memory_record.to_log_dict(), ensure_ascii=False),
                relevance_score=None,
            )
        elif not quality_allow:
            self.mysql_logger.log_memory(
                task_run_id=task_run_id,
                memory_key=f"task:{task_id}:write_skip_quality",
                operation_type="write_skip_quality",
                memory_content=json.dumps(memory_record.to_log_dict(), ensure_ascii=False),
                relevance_score=None,
            )
        elif not policy_allow:
            self.mysql_logger.log_memory(
                task_run_id=task_run_id,
                memory_key=f"task:{task_id}:write_skip_policy",
                operation_type="write_skip_policy",
                memory_content=json.dumps(memory_record.to_log_dict(), ensure_ascii=False),
                relevance_score=None,
            )
        t_memory_write_end = time.perf_counter()

        latency_ms = int((time.perf_counter() - wall_start) * 1000)
        policy_result = None
        if self.memory_policy is not None:
            policy_result = self.memory_policy.on_task_end(
                query=query,
                task_context=task_context,
                selected_memories=memory_items,
                final_answer=final_answer,
                memory_summary=memory_summary,
                strategy_note=strategy_note,
                memory_written=memory_written,
                latency_ms=latency_ms,
                task_id=task_id,
                task_order=task_order,
                support_task_ids=support_task_ids,
            )

        self.mysql_logger.update_task_result(
            task_run_id=task_run_id,
            final_answer=final_answer,
            success_flag=1,
            reward_score=policy_result.get("reward") if isinstance(policy_result, dict) else None,
            token_cost=None,
            latency_ms=latency_ms,
        )

        if self.knowledge_base is not None:
            self.knowledge_base.set_runtime_context(None)

        profile = {
            "retrieve_ms": round((t_retrieve_end - t_retrieve_start) * 1000, 3),
            "compose_ms": round((t_compose_end - t_retrieve_end) * 1000, 3),
            "agent_ms": round((t_agent_end - t_agent_start) * 1000, 3),
            "memory_write_ms": round((t_memory_write_end - t_memory_write_start) * 1000, 3),
            "total_ms": latency_ms,
            "memory_candidates": len(memory_items),
            "memory_written": memory_written,
            "raw_support_candidates": len(raw_support_memory_items),
            "filtered_support_candidates": len(support_memory_items),
        }
        self._maybe_log_profile(
            task_run_id=task_run_id,
            task_id=task_id,
            task_order=task_order,
            stream_id=stream_id,
            profile=profile,
        )

        return {
            "task_run_id": task_run_id,
            "task_id": task_id,
            "task_order": task_order,
            "stream_id": stream_id,
            "task_type": task_type,
            "entity": entity,
            "support_task_ids": support_task_ids,
            "query": query,
            "final_answer": final_answer,
            "memory_summary": memory_summary,
            "strategy_note": strategy_note,
            "latency_ms": latency_ms,
            "used_memories": memories,
            "policy_result": policy_result,
            "profile": profile,
        }

    def _filter_support_memories(self, support_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        过滤掉明显低质量的 support memory：
        1. 优先读结构化标签
        2. 向后兼容旧 memory bank 的文本规则
        """
        bad_patterns = [
            "未知",
            "未提供",
            "无法确定",
            "需进一步查询",
            "缺乏具体信息",
            "需要具体信息来确定",
            "需根据具体情况确定",
            "信息未说明",
            "尚不清楚",
        ]
        placeholder_tokens = ["#1", "#2", "#3", "#4", "#5"]

        filtered = []
        for item in support_items:
            if item.get("contains_placeholder") is True:
                continue
            if item.get("contains_unknown") is True:
                continue
            if item.get("memory_quality") == "reject":
                continue
            if item.get("gate_passed") is False:
                continue

            text = (item.get("memory_summary") or item.get("content") or "").strip()
            if not text:
                continue
            if any(p in text for p in bad_patterns):
                continue
            if any(tok in text for tok in placeholder_tokens):
                continue
            filtered.append(item)
        return filtered
