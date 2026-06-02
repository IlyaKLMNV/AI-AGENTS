"""qa_harness.pipeline — конвейер AI-поиска (step1 LLM-parse -> step2 payload -> step3 backend).

Доменно-специфичный, но отдельный от runners слой: extractor_agent, one_line, sourcing
импортируют отсюда (а не из CLI-раннера, как было в legacy).
"""

from .contract import validate_step1_contract
from .payload import build_step3_payload, make_base_payload
from .openai_step1 import PromptCfg, call_openai_step1, extract_response_text
from .backend_client import BackendCfg, call_backend_search_bool, classify_step3_error

__all__ = [
    "validate_step1_contract",
    "build_step3_payload",
    "make_base_payload",
    "PromptCfg",
    "call_openai_step1",
    "extract_response_text",
    "BackendCfg",
    "call_backend_search_bool",
    "classify_step3_error",
]
