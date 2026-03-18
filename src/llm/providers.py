from dataclasses import dataclass


@dataclass
class LLMConfig:
    provider: str  # "remote_api" / "local_vllm"
    model_name: str
    base_url: str
    api_key: str
