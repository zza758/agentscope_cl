from openai import OpenAI


def build_openai_client(provider: str, api_key: str, base_url: str) -> OpenAI:
    if provider == "local_vllm":
        return OpenAI(
            api_key=api_key or "EMPTY",
            base_url=base_url,
        )
    elif provider == "remote_api":
        return OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
