from birbal.models import *
from birbal.config import config

_provider = None


def get_ai_provider():
    global _provider
    if _provider is None:
        _provider = OllamaProvider()
    return _provider
