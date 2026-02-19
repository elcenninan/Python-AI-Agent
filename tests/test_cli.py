from abinitio_sql_agent.cli import _sanitize_argv


def test_sanitize_argv_drops_stray_backslash_tokens():
    argv = [
        "--schema",
        "schemas/example_schema.yaml",
        "\\",
        "--error",
        "failed",
        "\\",
    ]

    assert _sanitize_argv(argv) == ["--schema", "schemas/example_schema.yaml", "--error", "failed"]
