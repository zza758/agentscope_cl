import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from src.agents.main_agent import build_main_agent
from src.memory.embedder import DashScopeEmbedder
from src.memory.keyword_memory import KeywordMemoryManager
from src.memory.vector_memory import VectorMemoryManager
from src.pipeline.run_task import TaskRunner
from src.storage.mysql_logger import MySQLLogger
from src.tools.retrieval_tool import SimpleKnowledgeBase
from src.utils.config_loader import load_config, PROJECT_ROOT
from src.reranker.contrastive_reranker import ContrastiveReranker
from src.training.contrastive_infer import ContrastiveEncoderInfer


def load_tasks(tasks_file: str):
    import json
    tasks = []
    with open(tasks_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    tasks.sort(key=lambda x: x["task_order"])
    return tasks


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-file", type=str, required=True)
    parser.add_argument("--experiment-id", type=str, default=None)
    parser.add_argument("--task-ids", type=str, default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    ablation_cfg = config["ablation"]
    memory_cfg = config["memory"]
    kb_cfg = config["knowledge_base"]
    embedding_cfg = config.get("embedding", {})

    # 路径解析
    kb_path = Path(kb_cfg["kb_path"])
    if not kb_path.is_absolute():
        kb_path = PROJECT_ROOT / kb_path

    memory_path = Path(memory_cfg["storage_path"])
    if not memory_path.is_absolute():
        memory_path = PROJECT_ROOT / memory_path

    # 1. 初始化知识库
    kb = SimpleKnowledgeBase(
        kb_path=str(kb_path),
        default_top_k=kb_cfg.get("top_k", 3),
        score_threshold=kb_cfg.get("score_threshold"),
    )

    # 2. 初始化数据库日志器
    mysql_logger = MySQLLogger(config["database"])

    # 3. 绑定知识库日志器，并设置检索日志开关
    kb.bind_logger(mysql_logger)
    kb.set_logging_enabled(ablation_cfg.get("use_retrieval_logging", True))

    # 4. 初始化 memory manager（根据开关和后端选择）
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

    # 5. 根据开关决定是否给 Agent 注册知识库工具
    retrieval_func = kb.retrieve_knowledge if ablation_cfg.get("use_knowledge_base", True) else None

    agent = build_main_agent(
        model_config=config["model"],
        retrieval_func=retrieval_func,
        enable_kb_tool=ablation_cfg.get("use_knowledge_base", True),
    )

    # 6. 初始化任务运行器
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

    experiment_id = args.experiment_id or datetime.now().strftime("exp_%Y%m%d_%H%M%S")
    print(f"[Experiment] 当前实验ID: {experiment_id}")

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
    )

    # 7. 执行任务
    try:
        tasks = load_tasks(args.tasks_file)

        if args.task_ids:
            allow = {x.strip() for x in args.task_ids.split(",") if x.strip()}
            tasks = [t for t in tasks if t["task_id"] in allow]

        if args.max_tasks is not None:
            tasks = tasks[:args.max_tasks]

        for task in tasks:
            result = await runner.run_single_task(
                task_id=task["task_id"],
                task_order=task["task_order"],
                query=task["query"],
            )
            print(f"[{task['task_id']}]")
            print(result)
    finally:
        mysql_logger.close()


if __name__ == "__main__":
    asyncio.run(main())
