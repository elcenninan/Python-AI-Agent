# Ab Initio Log Recovery Agent (RAG)

This project was rebuilt as a focused **agentic RAG assistant** for Ab Initio graph failures.

You provide raw execution logs (like your `g6t_policy_trans_src` failure sample), and the agent:

1. Parses graph metadata + failed record fields.
2. Retrieves the closest schema/process definition from a schema knowledge base.
3. Produces a recommended SQL `UPDATE` to reset process status and related fields so rerun is possible.

## Why this helps when schema changes

Your schema/rules live in YAML knowledge docs (`schemas/*.yaml`).
When table names, status columns, or rerun codes change, update YAML instead of code.
The agent uses RAG retrieval over those docs to pick the right target behavior.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run with a log file

```bash
python -m abinitio_sql_agent.cli \
  --schema schemas/example_schema.yaml \
  --log-file sample.log
```

## Run with inline log text

```bash
python -m abinitio_sql_agent.cli \
  --schema schemas/example_schema.yaml \
  --log-data "Ab Initio Graph Execution Log ..."
```

## Schema knowledge format

```yaml
schemas:
  - name: g6t_policy_trans_src_prod
    table: G6T_POLICY_TRANS_STD
    process_status_column: process_status_code
    failed_value: FAILED
    rerun_ready_value: READY_FOR_RERUN
    id_columns: [trans_id, policy_id]
    mutable_columns: [process_status_code, retry_count, error_message, updated_by]
    notes: Main policy load schema
```

## Output

The CLI prints:

- SQL update template
- bind parameters extracted from failed record fields
- retrieval reasoning + matched schema docs
