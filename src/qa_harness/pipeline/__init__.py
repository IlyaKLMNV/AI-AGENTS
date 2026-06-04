"""qa_harness.pipeline — конвейер AI-поиска (step1 LLM-parse -> step2 payload -> step3 backend).

Доменно-специфичный, но отдельный от runners слой: extractor_agent, one_line, sourcing
импортируют отсюда (а не из CLI-раннера, как было в legacy).
"""

from .contract import validate_step1_contract
from .payload import build_step3_payload, make_base_payload, mapping_report
from .openai_step1 import PromptCfg, call_openai_step1, extract_response_text, parse_extractor_json
from .backend_client import BackendCfg, call_backend_search_bool, classify_step3_error

__all__ = [
    "validate_step1_contract",
    "build_step3_payload",
    "make_base_payload",
    "mapping_report",
    "PromptCfg",
    "call_openai_step1",
    "extract_response_text",
    "parse_extractor_json",
    "BackendCfg",
    "call_backend_search_bool",
    "classify_step3_error",
]
