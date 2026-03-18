import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from agentscope.message import Msg

from src.memory.memory_record import MemoryRecord
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

    def _retrieve_memories(self, query: str, task_context: TaskContext) -> List[Dict[str, Any]]:
        if not self.ablation_cfg.get("use_memory", True):
            return []
        if self.memory_manager is None:
            return []

        candidate_top_k = self.memory_top_k
        if self.contrastive_cfg.get("rerank_enabled", False):
            candidate_top_k = int(
                self.contrastive_cfg.get("candidate_top_k", self.memory_top_k)
            )

        memory_items = self.memory_manager.retrieve_memory_with_scores(
            query=query,
            task_context=task_context,
            top_k=candidate_top_k,
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
            memory_items = memory_items[: self.memory_top_k]

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
            parts = [f"[历史经验{idx}"]
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
            meta_text = f" ({'; '.join(meta)})" if meta else ""
            lines.append(f"{prefix}{meta_text} {item.get('content', '')}")

        return "\n".join(lines)

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

        start_time = time.time()
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

        if self.knowledge_base is not None and self.ablation_cfg.get("use_knowledge_base", True):
            self.knowledge_base.set_runtime_context(task_run_id)
        elif self.knowledge_base is not None:
            self.knowledge_base.set_runtime_context(None)

        memory_items = self._retrieve_memories(query=query, task_context=task_context)
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

        user_prompt = (
            f"当前任务:\n{query}\n\n"
            f"当前任务元数据:\n{task_meta_block}\n\n"
            f"可参考的历史经验:\n{formatted_memories}\n\n"
            f"请先判断是否需要调用知识检索工具，再完成任务。\n"
            f"最终输出请严格使用以下格式：\n\n"
            f"〖最终答案〗\n"
            f"写给用户看的最终回答。\n\n"
            f"〖记忆摘要〗\n"
            f"提炼出适合长期记忆检索的一段摘要。要求：\n"
            f"1. 只保留以后可复用的知识结论；\n"
            f"2. 不要写“无需检索”“根据历史经验”等过程性话语；\n"
            f"3. 尽量简洁、信息完整。\n\n"
            f"〖策略备注〗\n"
            f"简要说明这次任务处理策略，可用于后续策略学习。\n"
        )

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
        response = await self.agent(msg)
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

        self.mysql_logger.log_memory(
            task_run_id=task_run_id,
            memory_key=f"task:{task_id}:write_candidate",
            operation_type="write_candidate",
            memory_content=memory_record.to_log_text(),
            relevance_score=None,
        )

        should_write = True
        if self.memory_policy is not None:
            should_write = self.memory_policy.should_write_memory(
                query=query,
                task_context=task_context,
                final_answer=final_answer,
                memory_summary=memory_summary,
                strategy_note=strategy_note,
            )

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
                memory_content=memory_record.to_log_text(),
                relevance_score=None,
            )

        latency_ms = int((time.time() - start_time) * 1000)

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
        }
