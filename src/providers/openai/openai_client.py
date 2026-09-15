from openai import OpenAI

from src.config import Settings, get_settings


def build_openai_client(settings: Settings | None) -> OpenAI:
    resolved_settings = settings or get_settings()
    return OpenAI(api_key=resolved_settings.openai_api_key, base_url=resolved_settings.openai_base_url)
