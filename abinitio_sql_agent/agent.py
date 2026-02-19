from __future__ import annotations

import re
from pathlib import Path

import yaml

from .models import ParsedLog, RetrievedDoc, SQLRecommendation, SchemaDoc
from .rag_store import RAGStore


class SQLRecoveryAgent:
    """RAG-powered advisor that maps Ab Initio failures to rerun SQL updates."""

    def __init__(self, docs: list[SchemaDoc] | None = None) -> None:
        self.docs = docs or []
        self.store = RAGStore()

        for doc in self.docs:
            text = (
                f"name {doc.name} table {doc.table} "
                f"status_column {doc.process_status_column} "
                f"failed_value {doc.failed_value} rerun_ready {doc.rerun_ready_value} "
                f"id_columns {' '.join(doc.id_columns)} "
                f"mutable_columns {' '.join(doc.mutable_columns)} notes {doc.notes}"
            )
            self.store.add(
                key=doc.name,
                text=text,
                payload={
                    "name": doc.name,
                    "table": doc.table,
                    "status_column": doc.process_status_column,
                    "rerun_ready_value": doc.rerun_ready_value,
                    "failed_value": doc.failed_value,
                },
            )
        if self.docs:
            self.store.build()

    @classmethod
    def from_yaml(cls, schema_path: str | Path) -> "SQLRecoveryAgent":
        raw = yaml.safe_load(Path(schema_path).read_text())
        docs = [SchemaDoc(**entry) for entry in raw["schemas"]]
        return cls(docs)

    @staticmethod
    def parse_log(log_text: str) -> ParsedLog:
        parsed = ParsedLog()

        metadata_patterns = {
            "graph_name": r"Graph Name\s*:\s*(.+)",
            "project": r"Project\s*:\s*(.+)",
            "environment": r"Environment\s*:\s*(.+)",
            "status": r"Graph Status\s*:\s*(.+)",
        }
        for field, pattern in metadata_patterns.items():
            match = re.search(pattern, log_text)
            if match:
                setattr(parsed, field, match.group(1).strip())

        comp = re.search(r"Component\s+([^\(\n]+)", log_text)
        if comp:
            parsed.component = comp.group(1).strip()

        table = re.search(r"Loading records into table\s+([A-Za-z0-9_\.]+)", log_text)
        if table:
            parsed.table = table.group(1).strip()

        db_error = re.search(r"DB Error\s*:\s*(.+)", log_text)
        if db_error:
            parsed.db_error = db_error.group(1).strip()

        rejected = re.search(r"Rejected Records\s*:\s*(\d+)", log_text)
        if rejected:
            parsed.rejected_records = int(rejected.group(1))

        if "Failed Record:" in log_text:
            block = log_text.split("Failed Record:", 1)[1]
            fields: dict[str, str] = {}
            for line in block.splitlines():
                line = line.strip()
                if not line or line.startswith("[") or line.startswith("*") or ":" in line:
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    fields[key.strip()] = value.strip()
            parsed.failed_record = fields

        return parsed

    def recommend_sql(self, log_text: str) -> SQLRecommendation:
        if not self.docs:
            return self._recommend_sql_from_log_only(log_text)

        parsed = self.parse_log(log_text)
        query = " ".join(
            [
                parsed.graph_name,
                parsed.project,
                parsed.environment,
                parsed.component,
                parsed.table,
                parsed.db_error,
                " ".join(f"{k} {v}" for k, v in parsed.failed_record.items()),
            ]
        ).strip()

        retrieved = self.store.retrieve(query=query, top_k=3)
        if not retrieved:
            raise ValueError("No schema documents loaded for retrieval")

        top_chunk, top_score = retrieved[0]
        doc = next(item for item in self.docs if item.name == top_chunk.key)

        where_parts = [f"{doc.process_status_column} = :failed_value"]
        params: dict[str, str] = {
            "ready_value": doc.rerun_ready_value,
            "failed_value": doc.failed_value,
        }

        for id_col in doc.id_columns:
            if id_col in parsed.failed_record:
                where_parts.append(f"{id_col} = :{id_col}")
                params[id_col] = parsed.failed_record[id_col]

        set_parts = [f"{doc.process_status_column} = :ready_value"]
        for column in doc.mutable_columns:
            if column == doc.process_status_column:
                continue
            set_parts.append(f"{column} = :{column}")
            if "retry" in column.lower():
                params[column] = "1"
            elif "error" in column.lower():
                params[column] = "NULL"
            else:
                params[column] = "UPDATED_BY_AGENT"

        sql = (
            f"UPDATE {doc.table}\n"
            f"SET {', '.join(set_parts)}\n"
            f"WHERE {' AND '.join(where_parts)};"
        )

        reason = (
            f"Matched schema '{doc.name}' (score={top_score:.3f}) using RAG over schema docs. "
            f"Detected failed table '{parsed.table or doc.table}' with DB error '{parsed.db_error}'."
        )

        evidence = [
            RetrievedDoc(
                name=chunk.key,
                table=chunk.payload["table"],
                score=score,
                excerpt=chunk.text,
            )
            for chunk, score in retrieved
        ]

        return SQLRecommendation(
            table=doc.table,
            sql=sql,
            params=params,
            reason=reason,
            retrieved_docs=evidence,
        )

    def _recommend_sql_from_log_only(self, log_text: str) -> SQLRecommendation:
        parsed = self.parse_log(log_text)

        if not parsed.table:
            raise ValueError(
                "Unable to infer target table from log. Include a line like 'Loading records into table ...'."
            )

        identifier_columns = list(parsed.failed_record.keys())
        set_parts = ["process_status_code = :ready_value"]
        where_parts: list[str] = []
        params: dict[str, str] = {"ready_value": "READY_FOR_RERUN"}

        for column in identifier_columns:
            where_parts.append(f"{column} = :{column}")
            params[column] = parsed.failed_record[column]

        if not where_parts:
            where_parts = ["1 = 1"]

        sql = (
            f"UPDATE {parsed.table}\n"
            f"SET {', '.join(set_parts)}\n"
            f"WHERE {' AND '.join(where_parts)};"
        )

        return SQLRecommendation(
            table=parsed.table,
            sql=sql,
            params=params,
            reason=(
                "Generated SQL from log-only context without schema validation. "
                "The statement uses failed-record keys as row identifiers and avoids fixed value checks."
            ),
            retrieved_docs=[],
        )
