from abinitio_sql_agent.cli import _sanitize_argv, build_parser


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
    args = parser.parse_args(
        [
            "--schema",
            "schemas/example_schema.yaml",
            "--error",
            "failed",
        ]
    )

    assert args.log_data == "failed"


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
