# Ab Initio Failure SQL RAG Agent

This repository contains a lightweight **RAG-style AI agent** that generates safe SQL `UPDATE` statements when an Ab Initio job fails on Unix.

The goal is to reset failed records to a restartable status so downstream processing can continue.

Now the agent supports two generation modes:
- **Deterministic fallback template** (always available).
- **LangChain + LLM API generation** when `--llm-model` is provided and API credentials are configured.

## Is RAG the right approach?

Yes—**if you have multiple schemas, job-specific rules, and evolving operational notes**.

RAG helps by:
- Retrieving the most relevant table/schema/rules for the error.
- Grounding SQL generation in your own metadata (instead of hallucinating table names/columns).
- Letting operations teams update behavior by editing knowledge files, not code.

For very small static workflows (1-2 tables), a deterministic rule engine may be enough. This project supports both:
- Retrieval (`TF-IDF`) for selecting relevant schema chunks.
- Guardrailed SQL generation from retrieved metadata.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m abinitio_sql_agent.cli \
  --schema schemas/example_schema.yaml \
  --error "Ab Initio graph failed: duplicate key while loading orders" \
  --pk-column order_id \
  --pk-value ORD-1001

# Optional: use LLM generation (LangChain + OpenAI-compatible API)
python -m abinitio_sql_agent.cli \
  --schema schemas/example_schema.yaml \
  --error "Ab Initio graph failed: duplicate key while loading orders" \
  --pk-column order_id \
  --pk-value ORD-1001 \
  --llm-model gpt-4o-mini
```

## Inputs the agent expects

- One or more schema files (YAML) with:
  - table name
  - primary key columns
  - status column
  - optional retry/error timestamp columns
  - allowed status transitions
- Runtime failure context:
  - raw error text from Ab Initio logs
  - failing key info (pk column + value)

## Output

A structured response containing:
- selected table and confidence
- retrieved evidence snippets
- parameterized SQL update statement
- bind parameters
- operational explanation

## Next step for you

Share your actual table schemas + restart status logic, and this agent can be tuned to your production rules.
