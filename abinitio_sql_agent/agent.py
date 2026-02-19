from __future__ import annotations

import json
from pathlib import Path

import yaml

from .models import RetrievedChunk, SQLRecommendation, TableSchema
from .rag_store import RAGStore

try:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_ollama import ChatOllama
except ImportError:  # pragma: no cover - optional dependency at runtime
    ChatOllama = None
    ChatPromptTemplate = None
    StrOutputParser = None


class SQLUpdateAgent:
    def __init__(
        self,
        schemas: list[TableSchema],
        llm_model: str | None = None,
        ollama_base_url: str = "http://localhost:11434",
    ) -> None:
        self.schemas = {s.table: s for s in schemas}
        self.store = RAGStore()
        self.llm_model = llm_model
        self.ollama_base_url = ollama_base_url

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
    def from_yaml(
        cls,
        path: str | Path,
        llm_model: str | None = None,
        ollama_base_url: str = "http://localhost:11434",
    ) -> "SQLUpdateAgent":
        data = yaml.safe_load(Path(path).read_text())
        tables = [TableSchema(**entry) for entry in data["tables"]]
        return cls(tables, llm_model=llm_model, ollama_base_url=ollama_base_url)

    def _build_fallback_recommendation(
        self,
        schema: TableSchema,
        pk_column: str,
        pk_value: str,
        error_text: str,
        override_restart_status: str | None,
    ) -> tuple[str, dict[str, str], str]:
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
        explanation = (
            "Generated SQL with deterministic fallback template due to unavailable or invalid LLM output."
        )
        return sql, params, explanation

    def _build_llm_recommendation(
        self,
        schema: TableSchema,
        pk_column: str,
        pk_value: str,
        error_text: str,
        override_restart_status: str | None,
    ) -> tuple[tuple[str, dict[str, str], str] | None, str | None]:
        if not self.llm_model:
            return None, "LLM disabled because --llm-model was not provided."
        if ChatOllama is None or ChatPromptTemplate is None or StrOutputParser is None:
            return (
                None,
                "LLM dependencies are not installed. Install requirements including langchain-ollama.",
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You generate one safe SQL UPDATE statement for failed Ab Initio rows. "
                        "Return ONLY valid JSON with keys: sql, params, explanation. "
                        "Use bind placeholders like :pk_value. "
                        "Must include a guard where clause checking current failed status."
                    ),
                ),
                (
                    "human",
                    (
                        "Schema: {schema_json}\n"
                        "error_text: {error_text}\n"
                        "pk_column: {pk_column}\n"
                        "pk_value: {pk_value}\n"
                        "override_restart_status: {override_restart_status}\n"
                        "Build SQL + params for a restart update."
                    ),
                ),
            ]
        )

        chain = (
            prompt
            | ChatOllama(model=self.llm_model, temperature=0, base_url=self.ollama_base_url)
            | StrOutputParser()
        )
        try:
            raw = chain.invoke(
                {
                    "schema_json": json.dumps(schema.__dict__),
                    "error_text": error_text,
                    "pk_column": pk_column,
                    "pk_value": pk_value,
                    "override_restart_status": override_restart_status,
                }
            )
        except Exception as exc:  # pragma: no cover - network/model runtime issue
            return None, f"LLM invocation failed: {exc}"

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None, "LLM output was not valid JSON."
        if not isinstance(payload, dict):
            return None, "LLM output JSON was not an object."
        sql = payload.get("sql")
        params = payload.get("params")
        explanation = payload.get("explanation", "Generated by LLM")
        if not isinstance(sql, str) or not isinstance(params, dict):
            return None, "LLM output was missing a valid sql string or params object."
        if ":pk_value" not in sql or schema.failed_status not in sql:
            return None, "LLM SQL did not include required safety guards (:pk_value and failed_status check)."
        params.setdefault("pk_value", pk_value)
        params.setdefault("failed_status", schema.failed_status)
        return (sql, params, explanation), None

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

        result, llm_issue = self._build_llm_recommendation(
            schema=schema,
            pk_column=pk_column,
            pk_value=pk_value,
            error_text=error_text,
            override_restart_status=override_restart_status,
        )
        if result is None:
            sql, params, generation_explanation = self._build_fallback_recommendation(
                schema=schema,
                pk_column=pk_column,
                pk_value=pk_value,
                error_text=error_text,
                override_restart_status=override_restart_status,
            )
            if llm_issue:
                generation_explanation = f"{generation_explanation} Cause: {llm_issue}"
        else:
            sql, params, generation_explanation = result

        evidence = [
            RetrievedChunk(table=chunk.table, content=chunk.text, score=score)
            for chunk, score in retrieved
        ]

        explanation = (
            f"Selected table '{schema.table}' from retrieved schema context. "
            f"{generation_explanation}"
        )

        return SQLRecommendation(
            table=schema.table,
            sql=sql,
            params=params,
            evidence=evidence,
            explanation=explanation,
            confidence=best_score,
        )
