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
    def _infer_schema_from_log_heuristic(log_data: str) -> TableSchema:
        normalized = log_data.lower()
        table_match = re.search(r"\btable\s+([A-Za-z_][A-Za-z0-9_$#.]*)", log_data, flags=re.IGNORECASE)
        component_match = re.search(r"\bmp_[A-Za-z0-9_]*?(stg_[A-Za-z0-9_]+)", normalized)
        failed_fields = SQLUpdateAgent._extract_failed_record_fields(log_data)

        table_name = "stg_failed_records"
        if table_match:
            table_name = table_match.group(1)
        elif component_match:
            table_name = component_match.group(1)
        elif "orders" in normalized:
            table_name = "stg_orders"

        pk_column = "id"
        if failed_fields:
            first_key = next(iter(failed_fields.keys()))
            if first_key:
                pk_column = first_key

        status_column = "status"
        for candidate in ["record_status", "status", "status_flag"]:
            if re.search(rf"\b{candidate}\b", log_data, flags=re.IGNORECASE):
                status_column = candidate
                break

        return TableSchema(
            table=table_name,
            primary_keys=[pk_column],
            status_column=status_column,
            failed_status="FAILED",
            restart_status="READY_RETRY",
            notes="Inferred from runtime log data",
        )

    @classmethod
    def from_log(
        cls,
        log_data: str,
        llm_model: str | None = None,
        ollama_base_url: str = "http://localhost:11434",
    ) -> "SQLUpdateAgent":
        schema = cls._infer_schema_from_log_heuristic(log_data)
        return cls([schema], llm_model=llm_model, ollama_base_url=ollama_base_url)

    @staticmethod
    def _extract_pk_from_log(log_data: str, pk_column: str) -> str | None:
        patterns = [
            rf"\b{re.escape(pk_column)}\b\s*[:=]\s*['\"]?([A-Za-z0-9_.\-/]+)",
            rf"\b{re.escape(pk_column)}\b\s+is\s+['\"]?([A-Za-z0-9_.\-/]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, log_data, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip("'\" ")
        return None

    @staticmethod
    def _extract_failed_record_fields(log_data: str) -> dict[str, str]:
        failed_record_match = re.search(
            r"Failed\s+Record\s*:\s*(.+?)(?:\n\s*\n|\n\[[0-9]{2}:[0-9]{2}:[0-9]{2}\]|\Z)",
            log_data,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not failed_record_match:
            return {}

        segment = failed_record_match.group(1)
        flattened = " ".join(line.strip() for line in segment.splitlines()).strip()
        if not flattened:
            return {}

        fields: dict[str, str] = {}
        for part in re.split(r"\|", flattened):
            item = part.strip()
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            cleaned_key = key.strip()
            cleaned_value = value.strip().strip("'\"")
            if cleaned_key:
                fields[cleaned_key] = cleaned_value
        return fields

    @classmethod
    def _resolve_pk_inputs(
        cls,
        schema: TableSchema,
        log_data: str,
        extracted_fields: dict[str, str],
        pk_column: str | None,
        pk_value: str | None,
    ) -> tuple[str, str]:
        resolved_pk_column = (pk_column or (schema.primary_keys[0] if schema.primary_keys else "")).strip()
        if not resolved_pk_column:
            raise ValueError("No primary key column available in schema for SQL guard generation.")

        if pk_value:
            return resolved_pk_column, pk_value

        for field_name, field_value in extracted_fields.items():
            if field_name.lower() == resolved_pk_column.lower() and field_value:
                return resolved_pk_column, field_value

        extracted = cls._extract_pk_from_log(log_data, resolved_pk_column)
        if extracted:
            return resolved_pk_column, extracted

        raise ValueError(
            "Could not infer primary key value from log_data. "
            f"Include '{resolved_pk_column}=<value>' (or '{resolved_pk_column}: <value>') in the log, "
            "or pass --pk-value explicitly."
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
        return (sql, params, explanation), None

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
        extracted_fields = self._extract_failed_record_fields(log_data)
        retrieved = self.store.retrieve(log_data)

        if not retrieved:
            raise ValueError("No schema context found. Load at least one table schema.")

        schema, detected_status, selection_reason = self._select_schema_and_status(
            log_data=log_data,
            retrieved=retrieved,
            override_restart_status=override_restart_status,
        )
        resolved_pk_column, resolved_pk_value = self._resolve_pk_inputs(
            schema=schema,
            log_data=log_data,
            extracted_fields=extracted_fields,
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
            f"{selection_reason or ''} {generation_explanation}"
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
