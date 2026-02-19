from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SchemaDoc:
    """Single retrievable schema/process document used by the RAG index."""

    name: str
    table: str
    process_status_column: str
    rerun_ready_value: str
    failed_value: str = "FAILED"
    id_columns: list[str] = field(default_factory=list)
    mutable_columns: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class ParsedLog:
    graph_name: str = ""
    project: str = ""
    environment: str = ""
    component: str = ""
    table: str = ""
    db_error: str = ""
    status: str = ""
    rejected_records: int = 0
    failed_record: dict[str, str] = field(default_factory=dict)


@dataclass
class RetrievedDoc:
    name: str
    table: str
    score: float
    excerpt: str


@dataclass
class SQLRecommendation:
    table: str
    sql: str
    params: dict[str, str]
    reason: str
    retrieved_docs: list[RetrievedDoc] = field(default_factory=list)
