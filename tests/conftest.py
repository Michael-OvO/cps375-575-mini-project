import pytest


@pytest.fixture(autouse=True)
def _isolated_claude_ledger(tmp_path, monkeypatch):
    """Keep the Claude teacher's usage ledger out of the real ~/.claude/projects during tests."""
    monkeypatch.setenv("CLAUDE_TEACHER_LEDGER_DIR", str(tmp_path / "claude-ledger"))
    monkeypatch.delenv("CLAUDE_TEACHER_LEDGER", raising=False)
