from abinitio_sql_agent.cli import build_parser


def test_parser_accepts_log_file():
    parser = build_parser()
    args = parser.parse_args(["--schema", "schemas/example_schema.yaml", "--log-file", "sample.log"])

    assert args.log_file == "sample.log"
    assert args.log_data is None


def test_parser_accepts_inline_log_data():
    parser = build_parser()
    args = parser.parse_args(
        ["--schema", "schemas/example_schema.yaml", "--log-data", "Ab Initio Graph Execution Log"]
    )

    assert args.log_data.startswith("Ab Initio")
