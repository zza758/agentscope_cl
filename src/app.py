import argparse
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List

from src.agents.main_agent import build_main_agent
from src.benchmarks.common.task_cache import BenchmarkTaskCache
from src.benchmarks.common.task_loader import load_benchmark_tasks, select_benchmark_tasks
from src.memory.embedder import DashScopeEmbedder
from src.memory.keyword_memory import KeywordMemoryManager
from src.memory.vector_memory import VectorMemoryManager
from src.pipeline.run_task import TaskRunner
from src.policy.rl_policy import RLMemoryPolicy
from src.policy.rule_policy import RuleBasedMemoryPolicy
from src.reranker.contrastive_reranker import ContrastiveReranker
from src.storage.buffered_jsonl_logger import BufferedJSONLLogger
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
    target_dir = log_dir / experiment_id

    if backend == "buffered_jsonl":
        logger_cfg = config.get("logger", {})
        return BufferedJSONLLogger(
            str(target_dir),
            flush_every=int(logger_cfg.get("flush_every", 100)),
            flush_interval=float(logger_cfg.get("flush_interval", 1.0)),
        )

    return JSONLLogger(str(target_dir))


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


def build_contrastive_reranker(config, ablation_cfg):
    contrastive_cfg = config.get("contrastive", {})
    if not (
        contrastive_cfg.get("enabled", False)
        and contrastive_cfg.get("rerank_enabled", False)
        and ablation_cfg.get("use_contrastive_rerank", False)
    ):
        return contrastive_cfg, None

    model_dir = Path(contrastive_cfg["model_dir"])
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir

    infer_engine = ContrastiveEncoderInfer(
        model_dir=str(model_dir),
        max_length=int(contrastive_cfg.get("max_length", 128)),
        device=contrastive_cfg.get("device"),
    )
    return contrastive_cfg, ContrastiveReranker(infer_engine)


def build_runner_factory(
    config,
    ablation_cfg,
    memory_cfg,
    experiment_id: str,
    logger,
    kb,
    memory_manager,
    contrastive_cfg,
    contrastive_reranker,
    args,
    enable_kb_runtime_context: bool,
) -> Callable[[], TaskRunner]:
    def _factory():
        retrieval_func = kb.retrieve_knowledge if ablation_cfg.get("use_knowledge_base", True) else None

        if args.mode == "benchmark":
            runtime_model_config = config.get("llm", config["model"])
            benchmark_fast_agent = True
        else:
            runtime_model_config = config["model"]
            benchmark_fast_agent = False

        agent = build_main_agent(
            model_config=runtime_model_config,
            retrieval_func=retrieval_func,
            enable_kb_tool=ablation_cfg.get("use_knowledge_base", True),
            benchmark_fast=benchmark_fast_agent,
        )

        memory_policy = build_policy(args, memory_top_k=memory_cfg.get("top_k", 3))
        return TaskRunner(
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
            enable_profile=args.profile,
            enable_kb_runtime_context=enable_kb_runtime_context,
        )

    return _factory


async def run_one_stream(
    stream_id: str,
    stream_tasks: List[Dict],
    runner_factory: Callable[[], TaskRunner],
    sem: asyncio.Semaphore,
):
    async with sem:
        runner = runner_factory()
        results = []
        print(f"[StreamStart] stream={stream_id} tasks={len(stream_tasks)}", flush=True)

        for task in stream_tasks:
            try:
                result = await runner.run_single_task(task=task)
                results.append(result)
                print(
                    f"[{result['task_id']}] stream={result.get('stream_id')} "
                    f"order={result.get('task_order')} latency_ms={result.get('latency_ms')}",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[TaskError] stream={stream_id} task_id={task.get('task_id')} error={repr(e)}",
                    flush=True,
                )
                raise

        print(f"[StreamDone] stream={stream_id} finished_tasks={len(results)}", flush=True)
        return stream_id, results


async def run_streams_concurrently(
    stream_to_tasks: Dict[str, List[Dict]],
    runner_factory,
    max_concurrent_streams: int,
):
    sem = asyncio.Semaphore(max_concurrent_streams)
    jobs = [
        run_one_stream(stream_id, stream_tasks, runner_factory, sem)
        for stream_id, stream_tasks in stream_to_tasks.items()
    ]
    return await asyncio.gather(*jobs, return_exceptions=True)


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
    parser.add_argument(
        "--logger-backend",
        choices=["auto", "mysql", "jsonl", "buffered_jsonl"],
        default="buffered_jsonl",
    )
    parser.add_argument("--log-dir", type=str, default="outputs/benchmark_logs")
    parser.add_argument("--policy-backend", choices=["none", "rule", "rl"], default="none")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--max-concurrent-streams", type=int, default=4)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--disable-retrieval-logging", action="store_true")
    parser.add_argument("--use-contrastive-rerank", action="store_true")
    parser.add_argument("--disable-contrastive-rerank", action="store_true")
    parser.add_argument("--memory-backend", choices=["keyword", "vector"], default=None)
    parser.add_argument("--memory-top-k", type=int, default=None)
    args = parser.parse_args()

    task_file = args.task_file or args.tasks_file
    if not task_file:
        raise ValueError("必须提供 --task-file 或 --tasks-file")

    config = load_config(config_path=args.config)
    ablation_cfg = config["ablation"]
    memory_cfg = config["memory"]
    kb_cfg = config["knowledge_base"]

    # ===== runtime override: 避免为每一轮对照准备单独 config 文件 =====
    if args.use_contrastive_rerank:
        ablation_cfg["use_contrastive_rerank"] = True
        config.setdefault("contrastive", {})
        config["contrastive"]["enabled"] = True
        config["contrastive"]["rerank_enabled"] = True

    if args.disable_contrastive_rerank:
        ablation_cfg["use_contrastive_rerank"] = False

    if args.memory_backend is not None:
        memory_cfg["backend"] = args.memory_backend

    if args.memory_top_k is not None:
        memory_cfg["top_k"] = args.memory_top_k

    experiment_id = args.experiment_id or datetime.now().strftime("exp_%Y%m%d_%H%M%S")
    print(f"[Experiment] 当前实验ID: {experiment_id}", flush=True)

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

    enable_retrieval_logging = ablation_cfg.get("use_retrieval_logging", True)
    if args.disable_retrieval_logging:
        enable_retrieval_logging = False

    # 当前 retrieval_tool 依赖共享的 current_task_run_id。
    # 并行 stream 模式下，如果开启 retrieval logging，会发生并发覆盖。
    # 这一轮为了保证正确性，自动关闭。
    if args.mode == "benchmark" and args.max_concurrent_streams > 1 and enable_retrieval_logging:
        print(
            "[Warn] 并行 stream 模式下自动关闭 retrieval logging，以避免共享 runtime_context 冲突。",
            flush=True,
        )
        enable_retrieval_logging = False

    kb.set_logging_enabled(enable_retrieval_logging)
    enable_kb_runtime_context = enable_retrieval_logging

    memory_manager = build_memory_manager(config, ablation_cfg)
    contrastive_cfg, contrastive_reranker = build_contrastive_reranker(config, ablation_cfg)

    runner_factory = build_runner_factory(
        config=config,
        ablation_cfg=ablation_cfg,
        memory_cfg=memory_cfg,
        experiment_id=experiment_id,
        logger=logger,
        kb=kb,
        memory_manager=memory_manager,
        contrastive_cfg=contrastive_cfg,
        contrastive_reranker=contrastive_reranker,
        args=args,
        enable_kb_runtime_context=enable_kb_runtime_context,
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

            task_cache = BenchmarkTaskCache.from_tasks(tasks)
            print(
                f"[Run] mode={args.mode} dataset={args.dataset} "
                f"tasks={task_cache.num_tasks} streams={task_cache.num_streams} "
                f"max_concurrent_streams={args.max_concurrent_streams}",
                flush=True,
            )

            if args.max_concurrent_streams <= 1:
                runner = runner_factory()
                for task in task_cache.flatten():
                    result = await runner.run_single_task(task=task)
                    print(
                        f"[{result['task_id']}] stream={result.get('stream_id')} "
                        f"order={result.get('task_order')}",
                        flush=True,
                    )
                    print(result["final_answer"], flush=True)
                    print("-" * 80, flush=True)
            else:
                stream_results = await run_streams_concurrently(
                    stream_to_tasks=task_cache.stream_to_tasks,
                    runner_factory=runner_factory,
                    max_concurrent_streams=args.max_concurrent_streams,
                )

                finished_tasks = 0
                failed_streams = 0
                failed_details = []

                for item in stream_results:
                    if isinstance(item, Exception):
                        failed_streams += 1
                        failed_details.append(repr(item))
                        print(f"[StreamError] {repr(item)}", flush=True)
                        continue

                    stream_id, results = item
                    finished_tasks += len(results)
                    print(
                        f"[StreamSummary] stream={stream_id} finished_tasks={len(results)}",
                        flush=True,
                    )

                print(
                    f"[Done] benchmark parallel finished. "
                    f"streams={len(stream_results)} tasks={finished_tasks} failed_streams={failed_streams}",
                    flush=True,
                )

                if failed_details:
                    print("[FailedDetails]", flush=True)
                    for detail in failed_details:
                        print(detail, flush=True)

        else:
            tasks = load_legacy_tasks(task_file)
            if args.task_ids:
                allow = {x.strip() for x in args.task_ids.split(",") if x.strip()}
                tasks = [t for t in tasks if t["task_id"] in allow]
            if args.max_tasks is not None:
                tasks = tasks[: args.max_tasks]

            print(f"[Run] mode={args.mode} dataset={args.dataset} tasks={len(tasks)}", flush=True)

            runner = runner_factory()
            for task in tasks:
                result = await runner.run_single_task(task=task)
                print(
                    f"[{result['task_id']}] stream={result.get('stream_id')} "
                    f"order={result.get('task_order')}",
                    flush=True,
                )
                print(result["final_answer"], flush=True)
                print("-" * 80, flush=True)
    finally:
        logger.close()


if __name__ == "__main__":
    asyncio.run(main())
