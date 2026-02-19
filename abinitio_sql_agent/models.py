from dataclasses import dataclass, field
from typing import Any


@dataclass
class TableSchema:
    table: str
    primary_keys: list[str]
    status_column: str
    failed_status: str = "FAILED"
    restart_status: str = "READY_RETRY"
    retry_count_column: str | None = None
    last_error_column: str | None = None
    updated_at_column: str | None = None
    notes: str = ""


@dataclass
class RetrievedChunk:
    table: str
    content: str
    score: float


@dataclass
class SQLRecommendation:
    table: str
    sql: str
    params: dict[str, Any] = field(default_factory=dict)
    evidence: list[RetrievedChunk] = field(default_factory=list)
    explanation: str = ""
    confidence: float = 0.0
    detected_status: str = ""
