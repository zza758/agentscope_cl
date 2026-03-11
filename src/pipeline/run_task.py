import time
from datetime import datetime
from typing import Dict, Any, List

from agentscope.message import Msg
import re

from src.runtime.task_context import TaskContext
from src.memory.memory_record import MemoryRecord
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


def clean_final_answer(answer: str, expose_reasoning: bool = True) -> str:
    """
    根据配置决定是否保留中间推理痕迹。
    """
    if expose_reasoning:
        return answer.strip()

    lines = []
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append(line)
            continue
        if stripped.startswith("无需调用知识检索工具"):
            continue
        if stripped.startswith("需要调用知识检索工具"):
            continue
        if stripped.startswith("根据历史经验"):
            continue
        lines.append(line)

    cleaned = "\n".join(lines).strip()
    return cleaned if cleaned else answer.strip()


def clean_answer_for_memory(answer: str) -> str:
    """
    为长期 memory 清洗答案：
    只保留可复用的知识结论，去掉检索决策、历史经验说明、重排序说明等元话语。
    """
    if not answer:
        return answer

    drop_if_contains = [
        "根据历史经验",
        "当前任务是",
        "当前任务“",
        "当前任务\"",
        "无需额外检索",
        "无需重复检索",
        "无需调用知识检索工具",
        "无需再次调用知识检索工具",
        "需要调用知识检索工具",
        "必须依赖外部知识检索",
        "可直接基于历史经验",
        "直接基于历史经验进行",
        "可直接基于已有知识",
        "已有知识进行归纳",
        "历史经验已覆盖",
        "属于同一问题的深化表述",
        "属于对同一主题的深化",
        "无需新增事实性信息",
        "rerank=",
        "重排序得分",
        "说明其可靠性较强",
        "答案如下",
    ]

    cleaned_lines = []
    for line in answer.splitlines():
        stripped = line.strip()

        if not stripped:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue

        # 只要这一行里含有这些过程性短语，就整行删掉
        if any(token in stripped for token in drop_if_contains):
            continue

        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()

    # 去掉开头常见解释性前缀
    cleaned = re.sub(r"^根据检索到的知识[，,:：]?\s*", "", cleaned)
    cleaned = re.sub(r"^根据历史经验\d*[，,:：]?\s*", "", cleaned)
    cleaned = re.sub(r"^因此[，,:：]?\s*", "", cleaned)
    cleaned = re.sub(r"^可直接基于已有知识进行归纳[，,:：]?\s*", "", cleaned)
    cleaned = re.sub(r"^可直接基于历史经验进行更[^：:\n]*[：:]\s*", "", cleaned)

    # 压缩多余空行
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    return cleaned if cleaned else answer.strip()


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
            final_top_k = int(self.contrastive_cfg.get("final_top_k", self.memory_top_k))
            memory_items = self.contrastive_reranker.rerank(
                query=query,
                candidates=memory_items,
                top_k=final_top_k,
            )

        return memory_items

    @staticmethod
    def _format_memory_items(memory_items: List[Dict[str, Any]]) -> str:
        if not memory_items:
            return "暂无历史经验。"

        lines = []
        for idx, item in enumerate(memory_items, start=1):
            base_score = item.get("score")
            rerank_score = item.get("contrastive_score")

            parts = [f"[历史经验{idx}"]
            if base_score is not None:
                parts.append(f"base={base_score:.4f}")
            if rerank_score is not None:
                parts.append(f"rerank={rerank_score:.4f}")
            prefix = "|".join(parts) + "]"

            lines.append(f"{prefix} {item.get('content', '')}")

        return "\n".join(lines)

    async def run_single_task(self, task_id: str, task_order: int, query: str) -> Dict[str, Any]:
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

        user_prompt = (
            f"当前任务:\n{query}\n\n"
            f"可参考的历史经验:\n{formatted_memories}\n\n"
            f"请先判断是否需要调用知识检索工具，再完成任务。\n"
            f"最终输出请严格使用以下格式：\n\n"
            f"【最终答案】\n"
            f"写给用户看的最终回答。\n\n"
            f"【记忆摘要】\n"
            f"提炼出适合长期记忆检索的一段摘要。要求：\n"
            f"1. 只保留以后可复用的知识结论；\n"
            f"2. 不要写“无需检索”“根据历史经验”等过程性话语；\n"
            f"3. 尽量简洁、信息完整。\n\n"
            f"【策略备注】\n"
            f"简要说明这次任务处理策略，可用于后续策略学习。\n"
        )

        self.mysql_logger.log_trajectory(
            task_run_id=task_run_id,
            step_no=1,
            agent_name="MainAgent",
            action_type="compose_input",
            action_input={"query": query, "memories": memories},
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

        final_answer = clean_final_answer(
            raw_answer,
            expose_reasoning=self.ablation_cfg.get("expose_reasoning", True),
        )

        memory_answer = clean_answer_for_memory(final_answer)
        memory_record = MemoryRecord(
            experiment_id=self.experiment_id,
            task_id=task_id,
            task_order=task_order,
            query=query,
            answer_raw=final_answer,
            memory_summary=memory_summary,
            strategy_note=strategy_note,
            created_at=MemoryRecord.now_ts(),
        )

        self.mysql_logger.log_memory(
            task_run_id=task_run_id,
            memory_key=f"task:{task_id}:write",
            operation_type="write",
            memory_content=memory_record.to_log_text(),
            relevance_score=None,
        )

        experience = "本轮任务完成后，后续遇到相似问题可优先参考此答案结构与调用到的知识。"

        if (
                self.ablation_cfg.get("use_memory", True)
                and self.ablation_cfg.get("use_memory_write", True)
                and self.memory_manager is not None
        ):
            self.memory_manager.write_memory(memory_record)

            if self.ablation_cfg.get("use_memory_logging", True):
                self.mysql_logger.log_memory(
                    task_run_id=task_run_id,
                    memory_key=f"task:{task_id}:write",
                    operation_type="write",
                    memory_content=(
                        f"任务: {query}\n"
                        f"答案: {memory_answer}\n"
                        f"经验: {experience}"
                    ),
                    relevance_score=None,
                )

        latency_ms = int((time.time() - start_time) * 1000)
        self.mysql_logger.update_task_result(
            task_run_id=task_run_id,
            final_answer=final_answer,
            success_flag=1,
            reward_score=None,
            token_cost=None,
            latency_ms=latency_ms,
        )

        if self.knowledge_base is not None:
            self.knowledge_base.set_runtime_context(None)

        return {
            "task_run_id": task_run_id,
            "task_id": task_id,
            "query": query,
            "final_answer": final_answer,
            "latency_ms": latency_ms,
            "used_memories": memories,
        }
