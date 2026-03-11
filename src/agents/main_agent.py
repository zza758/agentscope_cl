from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import DashScopeChatModel
from agentscope.tool import Toolkit


def build_main_agent(model_config, retrieval_func=None, enable_kb_tool=True):
    toolkit = Toolkit()

    if enable_kb_tool and retrieval_func is not None:
        toolkit.register_tool_function(
            retrieval_func,
            func_description=(
                "用于检索外部知识库中与当前问题相关的文本片段。"
                "当你缺少事实信息、需要查找背景知识或需要补充证据时，调用这个工具。"
            ),
        )

    sys_prompt = (
        "你是一个具备知识检索与经验复用能力的智能体。"
        "如果提供了历史经验，请优先判断其是否足以支持回答。"
        "如果仍缺少事实依据，再考虑调用知识检索工具。"
        "回答应准确、简洁、可解释。"
    )

    agent = ReActAgent(
        name="MainAgent",
        sys_prompt=sys_prompt,
        model=DashScopeChatModel(
            model_name=model_config["model_name"],
            api_key=model_config["dashscope_api_key"],
            stream=model_config.get("stream", False),
            generate_kwargs={
                "temperature": model_config.get("temperature", 0.3),
                "max_tokens": model_config.get("max_tokens", 1500),
            },
        ),
        formatter=DashScopeChatFormatter(),
        toolkit=toolkit,
        memory=InMemoryMemory(),
    )
    return agent
