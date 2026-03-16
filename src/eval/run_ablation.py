import argparse
import asyncio
import json
from copy import deepcopy
from pathlib import Path

from src.app import load_tasks
from src.agents.main_agent import build_main_agent
from src.memory.embedder import DashScopeEmbedder
from src.memory.keyword_memory import KeywordMemoryManager
from src.memory.vector_memory import VectorMemoryManager
from src.pipeline.run_task import TaskRunner
from src.policy.rl_policy import RLMemoryPolicy
from src.policy.rule_policy import RuleBasedMemoryPolicy
from src.reranker.contrastive_reranker import ContrastiveReranker
from src.storage.mysql_logger import MySQLLogger
from src.tools.retrieval_tool import SimpleKnowledgeBase
from src.training.contrastive_infer import ContrastiveEncoderInfer
from src.utils.config_loader import PROJECT_ROOT, load_config


def build_memory_policy(config: dict):
    ablation_cfg = config["ablation"]
    memory_cfg = config["memory"]
    policy_cfg = config.get("memory_policy", {})

    if ablation_cfg.get("use_rl_policy", False):
        return RLMemoryPolicy(
            max_select_k=policy_cfg.get("max_select_k", memory_cfg.get("top_k", 3)),
            min_summary_len=policy_cfg.get("min_summary_len", 10),
            score_threshold=policy_cfg.get("score_threshold", 0.0),
        )

    if ablation_cfg.get("use_memory_policy", False):
        return RuleBasedMemoryPolicy(
            max_select_k=policy_cfg.get("max_select_k", memory_cfg.get("top_k", 3)),
            min_summary_len=policy_cfg.get("min_summary_len", 10),
        )

    return None


def build_runtime(config: dict, experiment_id: str):
    ablation_cfg = config["ablation"]
    memory_cfg = config["memory"]
    kb_cfg = config["knowledge_base"]
    embedding_cfg = config.get("embedding", {})

    kb_path = Path(kb_cfg["kb_path"])
    if not kb_path.is_absolute():
        kb_path = PROJECT_ROOT / kb_path

    memory_path = Path(memory_cfg["storage_path"])
    if not memory_path.is_absolute():
        memory_path = PROJECT_ROOT / memory_path

    kb = SimpleKnowledgeBase(
        kb_path=str(kb_path),
        default_top_k=kb_cfg.get("top_k", 3),
        score_threshold=kb_cfg.get("score_threshold"),
    )

    mysql_logger = MySQLLogger(config["database"])
    kb.bind_logger(mysql_logger)
    kb.set_logging_enabled(ablation_cfg.get("use_retrieval_logging", True))

    memory_manager = None
    if ablation_cfg.get("use_memory", True):
        backend = memory_cfg.get("backend", "keyword")
        if backend == "vector" and ablation_cfg.get("use_vector_memory", True):
            embedder = DashScopeEmbedder(
                api_key=config["model"]["dashscope_api_key"],
                model_name=embedding_cfg.get("model_name", "text-embedding-v4"),
                normalize=embedding_cfg.get("normalize", True),
            )
            memory_manager = VectorMemoryManager(
                storage_path=str(memory_path),
                embedder=embedder,
                default_top_k=memory_cfg.get("top_k", 3),
                persistent=ablation_cfg.get("use_memory_persistence", True),
                deduplicate=memory_cfg.get("deduplicate", True),
            )
        else:
            memory_manager = KeywordMemoryManager(
                storage_path=str(memory_path),
                default_top_k=memory_cfg.get("top_k", 3),
                persistent=ablation_cfg.get("use_memory_persistence", True),
                deduplicate=memory_cfg.get("deduplicate", True),
            )

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

    memory_policy = build_memory_policy(config)

    runner = TaskRunner(
        agent=agent,
        memory_manager=memory_manager,
        mysql_logger=mysql_logger,
        knowledge_base=kb,
        ablation_cfg=ablation_cfg,
        memory_top_k=memory_cfg.get("top_k", 3),
        contrastive_cfg=contrastive_cfg,
        contrastive_reranker=contrastive_reranker,
        experiment_id=experiment_id,
        memory_policy=memory_policy,
    )
    return runner, mysql_logger


async def run_setting(base_config: dict, tasks_file: str, experiment_id: str, setting_name: str, output_file: Path):
    config = deepcopy(base_config)

    config["ablation"]["use_memory_policy"] = False
    config["ablation"]["use_rl_policy"] = False

    if setting_name == "memory_off_rerank_off":
        config["ablation"]["use_memory"] = False
        config["ablation"]["use_contrastive_rerank"] = False
    elif setting_name == "memory_on_rerank_off":
        config["ablation"]["use_memory"] = True
        config["ablation"]["use_contrastive_rerank"] = False
    elif setting_name == "memory_on_rerank_on":
        config["ablation"]["use_memory"] = True
        config["ablation"]["use_contrastive_rerank"] = True
    elif setting_name == "memory_on_rerank_on_policy_on":
        config["ablation"]["use_memory"] = True
        config["ablation"]["use_contrastive_rerank"] = True
        config["ablation"]["use_memory_policy"] = True
    elif setting_name == "memory_on_rerank_on_rl_policy":
        config["ablation"]["use_memory"] = True
        config["ablation"]["use_contrastive_rerank"] = True
        config["ablation"]["use_rl_policy"] = True
    else:
        raise ValueError(f"unknown setting: {setting_name}")

    runner, mysql_logger = build_runtime(config=config, experiment_id=experiment_id)
    tasks = load_tasks(tasks_file)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for task in tasks:
                result = await runner.run_single_task(
                    task_id=task["task_id"],
                    task_order=task["task_order"],
                    query=task["query"],
                )
                result["setting_name"] = setting_name
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
    finally:
        mysql_logger.close()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-file", type=str, required=True)
    parser.add_argument("--experiment-prefix", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/eval")
    args = parser.parse_args()

    config = load_config()

    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root = output_root / args.experiment_prefix

    settings = [
        "memory_off_rerank_off",
        "memory_on_rerank_off",
        "memory_on_rerank_on",
        "memory_on_rerank_on_policy_on",
        "memory_on_rerank_on_rl_policy",
    ]

    for setting_name in settings:
        await run_setting(
            base_config=config,
            tasks_file=args.tasks_file,
            experiment_id=f"{args.experiment_prefix}_{setting_name}",
            setting_name=setting_name,
            output_file=output_root / f"{setting_name}.jsonl",
        )

    print(f"评测完成，输出目录: {output_root}")


if __name__ == "__main__":
    asyncio.run(main())
