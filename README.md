# Ab Initio Failure SQL RAG Agent

This repository contains a lightweight **RAG-style AI agent** that generates SQL `UPDATE` statements when an Ab Initio job fails on Unix.

The goal is to reset failed records to a restartable status so downstream processing can continue.

Now the agent supports three generation capabilities:
- **RAG retrieval** (`TF-IDF`) to shortlist schema candidates from log data.
- **LangChain + Ollama schema/table/status understanding** to interpret logs and select the best table + target restart status from retrieved chunks.
- **SQL generator** (LLM-first, deterministic fallback) to produce `UPDATE` statements.

## Is RAG the right approach?

Yes—**if you have multiple schemas, job-specific rules, and evolving operational notes**.

RAG helps by:
- Retrieving the most relevant table/schema/rules for the error.
- Grounding SQL generation in your own metadata (instead of hallucinating table names/columns).
- Letting operations teams update behavior by editing knowledge files, not code.

For very small static workflows (1-2 tables), a deterministic rule engine may be enough. This project supports both:
- Retrieval (`TF-IDF`) for selecting relevant schema chunks.
- SQL generation from retrieved metadata.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m abinitio_sql_agent.cli \
  --schema schemas/example_schema.yaml \
  --log-data "Ab Initio graph failed: duplicate key while loading orders" \
  --pk-column order_id \
  --pk-value ORD-1001

# Or pass only log data (schema is inferred from the log)
python -m abinitio_sql_agent.cli \
  --log-data "Ab Initio graph failed for order_id=ORD-1001: duplicate key while loading orders" \
  --llm-model mistral:7b

# Tip: if your shell pasted stray "\\" tokens, the CLI now ignores them.

# Optional: use local Ollama model generation
# Start Ollama first, then pull your model: ollama pull mistral:7b
python -m abinitio_sql_agent.cli \
  --schema schemas/example_schema.yaml \
  --log-data "Ab Initio graph failed: duplicate key while loading orders" \
  --pk-column order_id \
  --pk-value ORD-1001 \
  --llm-model mistral:7b \
  --ollama-base-url http://localhost:11434
```

## Inputs the agent expects

- Optional schema files (YAML) with:
  - table name
  - primary key columns
  - status column
  - optional retry/error timestamp columns
  - allowed status transitions
- Runtime failure context:
  - raw log data from Ab Initio logs (required)
  - failing key info (pk column + value), either passed explicitly or inferred by the LLM from log text

## Output

A structured response containing:
- selected table and confidence
- retrieved evidence snippets
- parameterized SQL update statement
- bind parameters
- operational explanation

## Next step for you

Share your actual table schemas + restart status logic, and this agent can be tuned to your production rules.
