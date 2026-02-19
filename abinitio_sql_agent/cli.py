from __future__ import annotations

import argparse
import json
import sys

from .agent import SQLUpdateAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate SQL update for Ab Initio failed records")
    parser.add_argument("--schema", required=True, help="Path to schema YAML")
    parser.add_argument("--error", required=True, help="Failure log/error text")
    parser.add_argument("--pk-column", required=True, help="Primary key column name")
    parser.add_argument("--pk-value", required=True, help="Primary key value")
    parser.add_argument("--new-status", default=None, help="Override restart status")
    parser.add_argument("--llm-model", default=None, help="Optional Ollama model name (ex: mistral:7b)")
    parser.add_argument("--ollama-base-url", default="http://localhost:11434", help="Ollama API base URL")
    return parser


def _sanitize_argv(argv: list[str]) -> list[str]:
    """Drop stray shell line-continuation tokens copied as literal args."""
    return [arg for arg in argv if arg != "\\"]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args(_sanitize_argv(sys.argv[1:]))

    agent = SQLUpdateAgent.from_yaml(
        args.schema,
        llm_model=args.llm_model,
        ollama_base_url=args.ollama_base_url,
    )
    recommendation = agent.recommend_update(
        error_text=args.error,
        pk_column=args.pk_column,
        pk_value=args.pk_value,
        override_restart_status=args.new_status,
    )

    print("=== SQL Recommendation ===")
    print(recommendation.sql)
    print("\n=== Parameters ===")
    print(json.dumps(recommendation.params, indent=2))
    print("\n=== Explanation ===")
    print(recommendation.explanation)
    print(f"confidence={recommendation.confidence:.3f}")


if __name__ == "__main__":
    main()
