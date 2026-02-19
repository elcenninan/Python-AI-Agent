from __future__ import annotations

import argparse
import json

from .agent import SQLRecoveryAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RAG agent that reads Ab Initio logs and recommends rerun SQL"
    )
    parser.add_argument("--schema", required=True, help="Path to schema knowledge YAML")
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional file containing raw Ab Initio graph execution log",
    )
    parser.add_argument(
        "--log-data",
        default=None,
        help="Raw log text (use instead of --log-file)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.log_file and not args.log_data:
        raise SystemExit("Provide one of --log-file or --log-data")

    log_text = args.log_data
    if args.log_file:
        with open(args.log_file, "r", encoding="utf-8") as handle:
            log_text = handle.read()

    agent = SQLRecoveryAgent.from_yaml(args.schema)
    recommendation = agent.recommend_sql(log_text)

    print("=== SQL Recommendation ===")
    print(recommendation.sql)
    print("\n=== Parameters ===")
    print(json.dumps(recommendation.params, indent=2))
    print("\n=== Reason ===")
    print(recommendation.reason)
    print("\n=== Retrieved Schema Docs ===")
    for item in recommendation.retrieved_docs:
        print(f"- {item.name} table={item.table} score={item.score:.3f}")


if __name__ == "__main__":
    main()
