"""Kumiho Memory - Universal memory provider for AI agents."""

__version__ = "0.5.3"

from kumiho_memory.redis_memory import RedisMemoryBuffer, _token_override_var as redis_token_override_var
from kumiho_memory.memory_manager import (
    AutoAssessFn,
    MemoryAssessResult,
    UniversalMemoryManager,
    get_memory_space,
)
from kumiho_memory.summarization import (
    AnthropicAdapter,
    EmbeddingAdapter,
    LLMAdapter,
    MemorySummarizer,
    OpenAICompatAdapter,
    OpenAICompatEmbeddingAdapter,
)
from kumiho_memory.privacy import PIIRedactor, CredentialDetectedError
from kumiho_memory.retry import RetryQueue
from kumiho_memory.dream_state import DreamState, MemoryAssessment, DreamStateStats
from kumiho_memory.graph_augmentation import GraphAugmentedRecall, GraphAugmentationConfig
from kumiho_memory.space_profiler import (
    SPACE_CLASSES,
    SpaceProfile,
    SpaceProfiler,
    SpaceSignals,
    get_space_profile,
)
from kumiho_memory.assessors import (
    DEFAULT_STORAGE_POLICY,
    EvidencePolicy,
    create_evidence_assessor,
    create_llm_assessor,
    grade_evidence,
    heuristic_prefilter,
)
from kumiho_memory.evidence import (
    CORROBORATED,
    DEFAULT_EVIDENCE_LEVEL,
    EVIDENCE_LEVELS,
    OFFICIAL,
    SINGLE_SOURCE,
    UNVERIFIED,
    evidence_tag,
    parse_evidence,
)
from kumiho_memory.evidence_rank import (
    EvidenceRankConfig,
    apply_evidence_weights,
    evidence_badge,
)

__all__ = [
    "__version__",
    "LLMAdapter",
    "EmbeddingAdapter",
    "OpenAICompatAdapter",
    "OpenAICompatEmbeddingAdapter",
    "AnthropicAdapter",
    "RedisMemoryBuffer",
    "AutoAssessFn",
    "MemoryAssessResult",
    "UniversalMemoryManager",
    "get_memory_space",
    "MemorySummarizer",
    "PIIRedactor",
    "CredentialDetectedError",
    "RetryQueue",
    "DreamState",
    "MemoryAssessment",
    "DreamStateStats",
    "GraphAugmentedRecall",
    "GraphAugmentationConfig",
    "DEFAULT_STORAGE_POLICY",
    "EvidencePolicy",
    "create_evidence_assessor",
    "create_llm_assessor",
    "grade_evidence",
    "heuristic_prefilter",
    "OFFICIAL",
    "CORROBORATED",
    "SINGLE_SOURCE",
    "UNVERIFIED",
    "EVIDENCE_LEVELS",
    "DEFAULT_EVIDENCE_LEVEL",
    "evidence_tag",
    "parse_evidence",
    "EvidenceRankConfig",
    "apply_evidence_weights",
    "evidence_badge",
    "SPACE_CLASSES",
    "SpaceProfile",
    "SpaceProfiler",
    "SpaceSignals",
    "get_space_profile",
]
