from abinitio_sql_agent.cli import _render_sql_sample, _sanitize_argv, _to_sql_literal, build_parser


def test_sanitize_argv_drops_stray_backslash_tokens():
    argv = [
        "--schema",
        "schemas/example_schema.yaml",
        "\\",
        "--log-data",
        "failed",
        "\\",
    ]

    assert _sanitize_argv(argv) == ["--schema", "schemas/example_schema.yaml", "--log-data", "failed"]


def test_parser_supports_error_alias_for_log_data():
    parser = build_parser()
    args = parser.parse_args(["--error", "failed"])

    assert args.log_data == "failed"


def test_parser_allows_omitting_schema():
    parser = build_parser()
    args = parser.parse_args(["--log-data", "failed for order_id=ORD-1"])

    assert args.schema is None


def test_parser_allows_omitting_pk_inputs():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--schema",
            "schemas/example_schema.yaml",
            "--log-data",
            "failed for order_id=ORD-1",
        ]
    )

    assert args.pk_column is None
    assert args.pk_value is None


def test_to_sql_literal_formats_common_types():
    assert _to_sql_literal(None) == "NULL"
    assert _to_sql_literal(True) == "TRUE"
    assert _to_sql_literal(3) == "3"
    assert _to_sql_literal("O'Reilly") == "'O''Reilly'"


def test_render_sql_sample_substitutes_available_params_only():
    sql = "UPDATE t SET a = :value, b = :missing WHERE id = :pk_value"
    rendered = _render_sql_sample(sql, {"value": "READY", "pk_value": "ORD-1"})

    assert rendered == "UPDATE t SET a = 'READY', b = :missing WHERE id = 'ORD-1'"
