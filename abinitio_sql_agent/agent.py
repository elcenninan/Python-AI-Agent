from __future__ import annotations

from pathlib import Path

import yaml

from .models import RetrievedChunk, SQLRecommendation, TableSchema
from .rag_store import RAGStore


class SQLUpdateAgent:
    def __init__(self, schemas: list[TableSchema]) -> None:
        self.schemas = {s.table: s for s in schemas}
        self.store = RAGStore()
        for schema in schemas:
            self.store.add(
                schema.table,
                (
                    f"table {schema.table} "
                    f"pk {' '.join(schema.primary_keys)} "
                    f"status {schema.status_column} "
                    f"failed_status {schema.failed_status} "
                    f"restart_status {schema.restart_status} "
                    f"notes {schema.notes}"
                ),
            )
        self.store.build()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SQLUpdateAgent":
        data = yaml.safe_load(Path(path).read_text())
        tables = [TableSchema(**entry) for entry in data["tables"]]
        return cls(tables)

    def recommend_update(
        self,
        error_text: str,
        pk_column: str,
        pk_value: str,
        override_restart_status: str | None = None,
    ) -> SQLRecommendation:
        retrieved = self.store.retrieve(error_text)

        if not retrieved:
            raise ValueError("No schema context found. Load at least one table schema.")

        best_chunk, best_score = retrieved[0]
        schema = self.schemas[best_chunk.table]

        restart_status = override_restart_status or schema.restart_status
        set_clauses = [f"{schema.status_column} = :restart_status"]

        params: dict[str, str] = {
            "restart_status": restart_status,
            "pk_value": pk_value,
            "failed_status": schema.failed_status,
            "error_text": error_text.strip()[:500],
        }

        if schema.retry_count_column:
            set_clauses.append(
                f"{schema.retry_count_column} = COALESCE({schema.retry_count_column}, 0) + 1"
            )

        if schema.last_error_column:
            set_clauses.append(f"{schema.last_error_column} = :error_text")

        if schema.updated_at_column:
            set_clauses.append(f"{schema.updated_at_column} = CURRENT_TIMESTAMP")

        sql = (
            f"UPDATE {schema.table}\n"
            f"SET {', '.join(set_clauses)}\n"
            f"WHERE {pk_column} = :pk_value\n"
            f"  AND {schema.status_column} = :failed_status;"
        )

        evidence = [
            RetrievedChunk(table=chunk.table, content=chunk.text, score=score)
            for chunk, score in retrieved
        ]

        explanation = (
            f"Selected table '{schema.table}' from retrieved schema context. "
            f"Generated guarded update that only changes rows currently in '{schema.failed_status}'."
        )

        return SQLRecommendation(
            table=schema.table,
            sql=sql,
            params=params,
            evidence=evidence,
            explanation=explanation,
            confidence=best_score,
        )
