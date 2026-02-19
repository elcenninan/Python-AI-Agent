from abinitio_sql_agent.agent import SQLRecoveryAgent


def test_parse_log_extracts_failed_record_fields():
    log = """Ab Initio Graph Execution Log
Graph Name      : g6t_policy_trans_src
Project         : POLICY_ETL
Environment     : PROD
[18:32:13] INFO  : Loading records into table G6T_POLICY_TRANS_STD
[18:32:14] ERROR : Component mp_load_policy_trans (Database Write)
                  DB Error  : ORA-00001: unique constraint violated
                  Failed Record:
                  trans_id=TRX900145
                  policy_id=POL778899
                  source_system=CORE
Graph Status    : FAILED
Rejected Records: 1
"""
    parsed = SQLRecoveryAgent.parse_log(log)

    assert parsed.graph_name == "g6t_policy_trans_src"
    assert parsed.table == "G6T_POLICY_TRANS_STD"
    assert parsed.failed_record["trans_id"] == "TRX900145"
    assert parsed.status == "FAILED"


def test_recommend_sql_uses_retrieved_schema(tmp_path):
    schema = tmp_path / "schema.yaml"
    schema.write_text(
        """
schemas:
  - name: g6t_policy_trans_src_prod
    table: G6T_POLICY_TRANS_STD
    process_status_column: process_status_code
    failed_value: FAILED
    rerun_ready_value: READY_FOR_RERUN
    id_columns: [trans_id, policy_id]
    mutable_columns: [process_status_code, retry_count, error_message]
    notes: Policy transaction load.
""".strip()
    )

    log = """Graph Name      : g6t_policy_trans_src
Project         : POLICY_ETL
Environment     : PROD
[18:32:13] INFO  : Loading records into table G6T_POLICY_TRANS_STD
DB Error  : ORA-00001: unique constraint violated
Failed Record:
trans_id=TRX900145
policy_id=POL778899
Graph Status    : FAILED
"""

    agent = SQLRecoveryAgent.from_yaml(schema)
    recommendation = agent.recommend_sql(log)

    assert "UPDATE G6T_POLICY_TRANS_STD" in recommendation.sql
    assert "process_status_code = :ready_value" in recommendation.sql
    assert recommendation.params["trans_id"] == "TRX900145"
    assert recommendation.params["policy_id"] == "POL778899"
    assert recommendation.retrieved_docs[0].name == "g6t_policy_trans_src_prod"
