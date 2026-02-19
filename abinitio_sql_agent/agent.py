from __future__ import annotations

import json
import re
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

    @staticmethod
    def _default_log_schema() -> TableSchema:
        return TableSchema(
            table="stg_failed_records",
            primary_keys=["id"],
            status_column="status",
            failed_status="FAILED",
            restart_status="READY_RETRY",
            notes="Default schema because LLM schema extraction was unavailable.",
        )

    @classmethod
    def _infer_schema_from_log_with_llm(
        cls,
        log_data: str,
        llm_model: str | None,
        ollama_base_url: str,
    ) -> tuple[TableSchema, str | None]:
        fallback = cls._default_log_schema()
        if not llm_model:
            return fallback, "Schema inference fallback used because --llm-model was not provided."
        if ChatOllama is None or ChatPromptTemplate is None or StrOutputParser is None:
            return fallback, "Schema inference fallback used because LLM dependencies are missing."

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You infer table schema metadata from an Ab Initio failure log. "
                        "Return ONLY valid JSON with keys: table, primary_keys, status_column, "
                        "failed_status, restart_status, notes."
                    ),
                ),
                (
                    "human",
                    "log_data: {log_data}",
                ),
            ]
        )
        chain = (
            prompt
            | ChatOllama(model=llm_model, temperature=0, base_url=ollama_base_url)
            | StrOutputParser()
        )
        try:
            payload = json.loads(chain.invoke({"log_data": log_data}))
            if not isinstance(payload, dict):
                return fallback, "Schema inference fallback used because LLM output was not a JSON object."
            return TableSchema(**payload), None
        except Exception as exc:  # pragma: no cover - runtime/model issue
            return fallback, f"Schema inference fallback used due to LLM issue: {exc}"

    @classmethod
    def from_log(
        cls,
        log_data: str,
        llm_model: str | None = None,
        ollama_base_url: str = "http://localhost:11434",
    ) -> "SQLUpdateAgent":
        schema, note = cls._infer_schema_from_log_with_llm(log_data, llm_model, ollama_base_url)
        agent = cls([schema], llm_model=llm_model, ollama_base_url=ollama_base_url)
        agent._log_schema_inference_note = note
        return agent

    def _infer_pk_value_with_llm(
        self,
        log_data: str,
        pk_column: str,
    ) -> tuple[str | None, str | None]:
        if not self.llm_model:
            return None, "Could not infer primary key value because --llm-model was not provided."
        if ChatOllama is None or ChatPromptTemplate is None or StrOutputParser is None:
            return None, "Could not infer primary key value because LLM dependencies are missing."

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "Extract the failed record primary key value from the log. "
                        "Return ONLY valid JSON with keys pk_value and reason. "
                        "Set pk_value to null if unavailable."
                    ),
                ),
                ("human", "pk_column: {pk_column}\nlog_data: {log_data}"),
            ]
        )
        chain = (
            prompt
            | ChatOllama(model=self.llm_model, temperature=0, base_url=self.ollama_base_url)
            | StrOutputParser()
        )
        try:
            payload = json.loads(chain.invoke({"pk_column": pk_column, "log_data": log_data}))
        except Exception as exc:  # pragma: no cover - runtime/model issue
            return None, f"Could not infer primary key value due to LLM issue: {exc}"

        if not isinstance(payload, dict):
            return None, "Could not infer primary key value because LLM output was not a JSON object."
        value = payload.get("pk_value")
        reason = payload.get("reason")
        if isinstance(value, str) and value.strip():
            return value.strip(), reason if isinstance(reason, str) else None
        return None, reason if isinstance(reason, str) else "Could not infer primary key value from log_data."

    def _resolve_pk_inputs(
        self,
        schema: TableSchema,
        log_data: str,
        pk_column: str | None,
        pk_value: str | None,
    ) -> tuple[str, str, str | None]:
        resolved_pk_column = (pk_column or (schema.primary_keys[0] if schema.primary_keys else "")).strip()
        if not resolved_pk_column:
            raise ValueError("No primary key column available in schema for SQL guard generation.")

        if pk_value:
            return resolved_pk_column, pk_value, None

        llm_value, llm_reason = self._infer_pk_value_with_llm(
            log_data=log_data,
            pk_column=resolved_pk_column,
        )
        if llm_value:
            return resolved_pk_column, llm_value, llm_reason

        raise ValueError(
            (llm_reason or "Could not infer primary key value from log_data.")
            + f" Pass --pk-value explicitly for primary key column '{resolved_pk_column}'."
        )

    def _build_fallback_recommendation(
        self,
        schema: TableSchema,
        pk_column: str,
        pk_value: str,
        log_data: str,
        override_restart_status: str | None,
    ) -> tuple[str, dict[str, str], str]:
        restart_status = override_restart_status or schema.restart_status
        set_clauses = [f"{schema.status_column} = :restart_status"]

        params: dict[str, str] = {
            "restart_status": restart_status,
            "pk_value": pk_value,
            "failed_status": schema.failed_status,
            "error_text": log_data.strip()[:500],
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
        log_data: str,
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
                        "log_data: {log_data}\n"
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
                    "log_data": log_data,
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
        params.setdefault("pk_value", pk_value)
        params.setdefault("failed_status", schema.failed_status)

        validation_issue = self._validate_llm_sql(schema=schema, sql=sql)
        if validation_issue:
            return None, validation_issue
        return (sql, params, explanation), None

    @staticmethod
    def _validate_llm_sql(schema: TableSchema, sql: str) -> str | None:
        normalized = " ".join(sql.strip().split())
        if not normalized.lower().startswith("update "):
            return "LLM SQL validation failed: statement is not an UPDATE."

        table_match = re.match(r"(?is)^update\s+([\w\.\"]+)\s+set\s+", normalized)
        if not table_match:
            return "LLM SQL validation failed: could not parse UPDATE target table."

        table_token = table_match.group(1).strip('"')
        if table_token.lower() != schema.table.lower():
            return (
                "LLM SQL validation failed: UPDATE table does not match selected schema "
                f"('{table_token}' != '{schema.table}')."
            )

        set_match = re.search(r"(?is)\bset\b\s+(.*?)\s*(?:\bwhere\b|;|$)", normalized)
        if not set_match:
            return "LLM SQL validation failed: missing SET clause."

        allowed_columns = {
            schema.status_column.lower(),
            *(col.lower() for col in [schema.retry_count_column, schema.last_error_column, schema.updated_at_column] if col),
        }
        assignments = [part.strip() for part in set_match.group(1).split(",") if part.strip()]
        if not assignments:
            return "LLM SQL validation failed: SET clause has no assignments."

        for assignment in assignments:
            lhs = assignment.split("=", 1)[0].strip().strip('"')
            if lhs.lower() not in allowed_columns:
                return (
                    "LLM SQL validation failed: SET clause references unknown column "
                    f"'{lhs}' for table '{schema.table}'."
                )

        return None

    def _select_schema_and_status(
        self,
        log_data: str,
        retrieved: list[tuple],
        override_restart_status: str | None,
    ) -> tuple[TableSchema, str, str | None]:
        best_chunk, _best_score = retrieved[0]
        default_schema = self.schemas[best_chunk.table]
        default_status = override_restart_status or default_schema.restart_status

        if not self.llm_model:
            return default_schema, default_status, "LLM selector disabled; used top RAG chunk."
        if ChatOllama is None or ChatPromptTemplate is None or StrOutputParser is None:
            return default_schema, default_status, "LLM selector unavailable due to missing dependencies."

        candidates = []
        for chunk, score in retrieved:
            schema = self.schemas[chunk.table]
            candidates.append(
                {
                    "table": schema.table,
                    "status_column": schema.status_column,
                    "failed_status": schema.failed_status,
                    "restart_status": schema.restart_status,
                    "notes": schema.notes,
                    "score": score,
                }
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You identify which table should be updated and what target status to set, "
                        "grounded strictly in retrieved candidates. "
                        "Return ONLY JSON with keys table, target_status, reason."
                    ),
                ),
                (
                    "human",
                    (
                        "log_data: {log_data}\n"
                        "override_restart_status: {override_restart_status}\n"
                        "candidates: {candidates_json}\n"
                        "Choose one candidate table and target status."
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
                    "log_data": log_data,
                    "override_restart_status": override_restart_status,
                    "candidates_json": json.dumps(candidates),
                }
            )
            payload = json.loads(raw)
        except Exception as exc:  # pragma: no cover - runtime/model issue
            return default_schema, default_status, f"LLM selector failed: {exc}"

        table = payload.get("table") if isinstance(payload, dict) else None
        target_status = payload.get("target_status") if isinstance(payload, dict) else None
        reason = payload.get("reason") if isinstance(payload, dict) else None

        if table not in self.schemas:
            return default_schema, default_status, "LLM selector returned unknown table; used top RAG chunk."

        schema = self.schemas[table]
        selected_status = override_restart_status or (target_status if isinstance(target_status, str) and target_status.strip() else schema.restart_status)
        return schema, selected_status, reason if isinstance(reason, str) else None

    def recommend_update(
        self,
        log_data: str,
        pk_column: str | None = None,
        pk_value: str | None = None,
        override_restart_status: str | None = None,
    ) -> SQLRecommendation:
        retrieved = self.store.retrieve(log_data)

        if not retrieved:
            raise ValueError("No schema context found. Load at least one table schema.")

        schema, detected_status, selection_reason = self._select_schema_and_status(
            log_data=log_data,
            retrieved=retrieved,
            override_restart_status=override_restart_status,
        )
        resolved_pk_column, resolved_pk_value, pk_reason = self._resolve_pk_inputs(
            schema=schema,
            log_data=log_data,
            pk_column=pk_column,
            pk_value=pk_value,
        )

        result, llm_issue = self._build_llm_recommendation(
            schema=schema,
            pk_column=resolved_pk_column,
            pk_value=resolved_pk_value,
            log_data=log_data,
            override_restart_status=detected_status,
        )
        if result is None:
            sql, params, generation_explanation = self._build_fallback_recommendation(
                schema=schema,
                pk_column=resolved_pk_column,
                pk_value=resolved_pk_value,
                log_data=log_data,
                override_restart_status=detected_status,
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
            f"Target status '{detected_status}'. "
            f"{selection_reason or ''} {pk_reason or ''} {getattr(self, '_log_schema_inference_note', '')} {generation_explanation}"
        ).strip()

        return SQLRecommendation(
            table=schema.table,
            sql=sql,
            params=params,
            evidence=evidence,
            explanation=explanation,
            confidence=max(score for _, score in retrieved),
            detected_status=detected_status,
        )
