import argparse
import asyncio
from datetime import datetime
from pathlib import Path

from src.agents.main_agent import build_main_agent
from src.benchmarks.common.task_loader import load_benchmark_tasks, select_benchmark_tasks
from src.memory.embedder import DashScopeEmbedder
from src.memory.keyword_memory import KeywordMemoryManager
from src.memory.vector_memory import VectorMemoryManager
from src.pipeline.run_task import TaskRunner
from src.policy.rl_policy import RLMemoryPolicy
from src.policy.rule_policy import RuleBasedMemoryPolicy
from src.reranker.contrastive_reranker import ContrastiveReranker
from src.storage.jsonl_logger import JSONLLogger
from src.storage.mysql_logger import MySQLLogger
from src.tools.retrieval_tool import SimpleKnowledgeBase
from src.training.contrastive_infer import ContrastiveEncoderInfer
from src.utils.config_loader import PROJECT_ROOT, load_config


def load_legacy_tasks(tasks_file: str):
    import json

    tasks = []
    with open(tasks_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    tasks.sort(key=lambda x: x["task_order"])
    return tasks


def build_logger(args, config, experiment_id: str):
    backend = args.logger_backend
    db_cfg = config.get("database", {})

    if backend == "auto":
        backend = "mysql" if db_cfg.get("enabled", False) else "jsonl"

    if backend == "mysql":
        return MySQLLogger(db_cfg)

    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = PROJECT_ROOT / log_dir
    return JSONLLogger(str(log_dir / experiment_id))


def build_memory_manager(config, ablation_cfg):
    memory_cfg = config["memory"]
    embedding_cfg = config.get("embedding", {})

    memory_path = Path(memory_cfg["storage_path"])
    if not memory_path.is_absolute():
        memory_path = PROJECT_ROOT / memory_path

    if not ablation_cfg.get("use_memory", True):
        return None

    backend = memory_cfg.get("backend", "keyword")
    if backend == "vector" and ablation_cfg.get("use_vector_memory", True):
        embedder = DashScopeEmbedder(
            api_key=config["model"]["dashscope_api_key"],
            model_name=embedding_cfg.get("model_name", "text-embedding-v4"),
            normalize=embedding_cfg.get("normalize", True),
        )
        return VectorMemoryManager(
            storage_path=str(memory_path),
            embedder=embedder,
            default_top_k=memory_cfg.get("top_k", 3),
            persistent=ablation_cfg.get("use_memory_persistence", True),
            deduplicate=memory_cfg.get("deduplicate", True),
        )

    return KeywordMemoryManager(
        storage_path=str(memory_path),
        default_top_k=memory_cfg.get("top_k", 3),
        persistent=ablation_cfg.get("use_memory_persistence", True),
        deduplicate=memory_cfg.get("deduplicate", True),
    )


def build_policy(args, memory_top_k: int):
    if args.policy_backend == "rule":
        return RuleBasedMemoryPolicy(max_select_k=memory_top_k)
    if args.policy_backend == "rl":
        return RLMemoryPolicy(max_select_k=memory_top_k)
    return None


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["standard", "benchmark"], default="standard")
    parser.add_argument("--dataset", choices=["hotpotqa", "musique"], default=None)
    parser.add_argument("--task-file", type=str, default=None)
    parser.add_argument("--tasks-file", type=str, default=None)
    parser.add_argument("--experiment-id", type=str, default=None)
    parser.add_argument("--task-ids", type=str, default=None)
    parser.add_argument("--stream-ids", type=str, default=None)
    parser.add_argument("--stream-limit", type=int, default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--logger-backend", choices=["auto", "mysql", "jsonl"], default="jsonl")
    parser.add_argument("--log-dir", type=str, default="outputs/benchmark_logs")
    parser.add_argument("--policy-backend", choices=["none", "rule", "rl"], default="none")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    task_file = args.task_file or args.tasks_file
    if not task_file:
        raise ValueError("必须提供 --task-file 或 --tasks-file")

    config = load_config(config_path=args.config)
    ablation_cfg = config["ablation"]
    memory_cfg = config["memory"]
    kb_cfg = config["knowledge_base"]

    experiment_id = args.experiment_id or datetime.now().strftime("exp_%Y%m%d_%H%M%S")
    print(f"[Experiment] 当前实验ID: {experiment_id}")

    kb_path = Path(kb_cfg["kb_path"])
    if not kb_path.is_absolute():
        kb_path = PROJECT_ROOT / kb_path

    kb = SimpleKnowledgeBase(
        kb_path=str(kb_path),
        default_top_k=kb_cfg.get("top_k", 3),
        score_threshold=kb_cfg.get("score_threshold"),
    )

    logger = build_logger(args, config, experiment_id=experiment_id)
    kb.bind_logger(logger)
    kb.set_logging_enabled(ablation_cfg.get("use_retrieval_logging", True))

    memory_manager = build_memory_manager(config, ablation_cfg)

    retrieval_func = kb.retrieve_knowledge if ablation_cfg.get("use_knowledge_base", True) else None
    agent = build_main_agent(
        model_config=config["model"],
        retrieval_func=retrieval_func,
        enable_kb_tool=ablation_cfg.get("use_knowledge_base", True),
    )

    contrastive_cfg = config.get("contrastive", {})
    contrastive_reranker = None
    if (
        contrastive_cfg.get("enabled", False)
        and contrastive_cfg.get("rerank_enabled", False)
        and ablation_cfg.get("use_contrastive_rerank", False)
    ):
        model_dir = Path(contrastive_cfg["model_dir"])
        if not model_dir.is_absolute():
            model_dir = PROJECT_ROOT / model_dir
        infer_engine = ContrastiveEncoderInfer(
            model_dir=str(model_dir),
            max_length=int(contrastive_cfg.get("max_length", 128)),
        )
        contrastive_reranker = ContrastiveReranker(infer_engine)

    memory_policy = build_policy(args, memory_top_k=memory_cfg.get("top_k", 3))

    runner = TaskRunner(
        agent=agent,
        memory_manager=memory_manager,
        mysql_logger=logger,
        knowledge_base=kb,
        ablation_cfg=ablation_cfg,
        memory_top_k=memory_cfg.get("top_k", 3),
        contrastive_cfg=contrastive_cfg,
        contrastive_reranker=contrastive_reranker,
        memory_policy=memory_policy,
        experiment_id=experiment_id,
    )

    try:
        if args.mode == "benchmark":
            tasks = load_benchmark_tasks(task_file=task_file, dataset=args.dataset)
            tasks = select_benchmark_tasks(
                tasks,
                task_ids=args.task_ids.split(",") if args.task_ids else None,
                stream_ids=args.stream_ids.split(",") if args.stream_ids else None,
                max_tasks=args.max_tasks,
                stream_limit=args.stream_limit,
            )
        else:
            tasks = load_legacy_tasks(task_file)
            if args.task_ids:
                allow = {x.strip() for x in args.task_ids.split(",") if x.strip()}
                tasks = [t for t in tasks if t["task_id"] in allow]
            if args.max_tasks is not None:
                tasks = tasks[: args.max_tasks]

        print(f"[Run] mode={args.mode} dataset={args.dataset} tasks={len(tasks)}")

        for task in tasks:
            task_dict = task.to_dict() if hasattr(task, "to_dict") else task
            result = await runner.run_single_task(task=task_dict)
            print(f"[{result['task_id']}] stream={result.get('stream_id')} order={result.get('task_order')}")
            print(result["final_answer"])
            print("-" * 80)
    finally:
        logger.close()


if __name__ == "__main__":
    asyncio.run(main())
