from abinitio_sql_agent.agent import SQLUpdateAgent


def test_recommend_update_builds_guarded_sql(tmp_path):
    schema = tmp_path / "schema.yaml"
    schema.write_text(
        """
tables:
  - table: stg_orders
    primary_keys: [order_id]
    status_column: record_status
    failed_status: FAILED
    restart_status: READY_RETRY
    retry_count_column: retry_count
    last_error_column: last_error_message
    updated_at_column: updated_at
    notes: "orders"
""".strip()
    )

    agent = SQLUpdateAgent.from_yaml(schema)
    rec = agent.recommend_update(
        error_text="abinitio failed while loading orders",
        pk_column="order_id",
        pk_value="ORD-1",
    )

    assert "UPDATE stg_orders" in rec.sql
    assert "AND record_status = :failed_status" in rec.sql
    assert rec.params["restart_status"] == "READY_RETRY"
    assert rec.table == "stg_orders"
