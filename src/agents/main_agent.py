from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeChatFormatter, OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import DashScopeChatModel, OpenAIChatModel
from agentscope.tool import Toolkit


def build_main_agent(
    model_config,
    retrieval_func=None,
    enable_kb_tool=True,
    benchmark_fast: bool = False,
):
    toolkit = Toolkit()

    if enable_kb_tool and retrieval_func is not None:
        toolkit.register_tool_function(
            retrieval_func,
            func_description=(
                "用于检索外部知识库中与当前问题相关的文本片段。"
                "当你缺少事实信息、需要查找背景知识或需要补充证据时，调用这个工具。"
            ),
        )

    # benchmark_fast 只做低风险加速：
    # 1. sys_prompt 更短
    # 2. temperature 更稳
    # 3. max_tokens 更小
    if benchmark_fast:
        sys_prompt = (
            "你是一个具备知识检索与经验复用能力的智能体。"
            "若历史经验已足够，优先直接回答；"
            "仅在缺少事实依据时调用知识检索工具。"
            "回答应准确、简洁。"
        )
        default_temperature = 0.1
        default_max_tokens = 384
    else:
        sys_prompt = (
            "你是一个具备知识检索与经验复用能力的智能体。"
            "如果提供了历史经验，请优先判断其是否足以支持回答。"
            "如果仍缺少事实依据，再考虑调用知识检索工具。"
            "回答应准确、简洁、可解释。"
        )
        default_temperature = 0.3
        default_max_tokens = 1500

    provider = model_config.get("provider", "dashscope")

    if provider in {"local_vllm", "vllm", "openai", "openai_compatible"}:
        model = OpenAIChatModel(
            model_name=model_config["model_name"],
            api_key=model_config.get("api_key", "EMPTY"),
            client_kwargs={
                "base_url": model_config["base_url"],
            },
            stream=model_config.get("stream", False),
            generate_kwargs={
                "temperature": model_config.get("temperature", default_temperature),
                "max_tokens": model_config.get("max_tokens", default_max_tokens),
            },
        )
        formatter = OpenAIChatFormatter()
    elif provider == "dashscope":
        model = DashScopeChatModel(
            model_name=model_config["model_name"],
            api_key=model_config["dashscope_api_key"],
            stream=model_config.get("stream", False),
            generate_kwargs={
                "temperature": model_config.get("temperature", default_temperature),
                "max_tokens": model_config.get("max_tokens", default_max_tokens),
            },
        )
        formatter = DashScopeChatFormatter()
    else:
        raise ValueError(f"Unsupported model provider: {provider}")

    agent = ReActAgent(
        name="MainAgent",
        sys_prompt=sys_prompt,
        model=model,
        formatter=formatter,
        toolkit=toolkit,
        memory=InMemoryMemory(),
    )
    return agent
