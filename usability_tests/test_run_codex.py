"""Regression tests for Codex JSONL bookkeeping."""

from usability_tests.run_codex import cost, summarise_codex


def test_stream_items_are_counted_once_and_executions_keep_start_order():
    events = [
        {"type": "turn.started"},
        {"type": "item.started", "item": {
            "id": "slow", "type": "command_execution",
            "command": "/bin/zsh -lc 'python out/T1/solution.py'",
            "exit_code": None, "status": "in_progress",
        }},
        {"type": "item.started", "item": {
            "id": "fast", "type": "command_execution",
            "command": "/bin/zsh -lc 'brainsurgery out/T1/plan.yaml'",
            "exit_code": None, "status": "in_progress",
        }},
        {"type": "item.completed", "item": {
            "id": "fast", "type": "command_execution",
            "command": "/bin/zsh -lc 'brainsurgery out/T1/plan.yaml'",
            "exit_code": 0, "status": "completed", "aggregated_output": "ok",
        }},
        {"type": "item.started", "item": {"id": "edit", "type": "file_change"}},
        {"type": "item.completed", "item": {"id": "edit", "type": "file_change"}},
        {"type": "item.completed", "item": {
            "id": "slow", "type": "command_execution",
            "command": "/bin/zsh -lc 'python out/T1/solution.py'",
            "exit_code": 1, "status": "failed", "aggregated_output": "bad",
        }},
        {"type": "turn.completed", "usage": {
            "input_tokens": 10, "cached_input_tokens": 3,
            "cache_write_input_tokens": 2, "output_tokens": 2,
        }},
    ]

    summary = summarise_codex(events, "done")

    assert summary["tool_calls"] == 3
    assert summary["executions"] == 2
    assert summary["failed_executions"][0]["n"] == 1
    assert summary["first_execution_success"] is False
    assert summary["executions_until_first_success"] == 2
    assert summary["tokens_in"] == 5
    assert summary["cache_read_tokens"] == 3
    assert summary["cache_write_tokens"] == 2


def test_cost_uses_separate_cache_rates():
    summary = {
        "tokens_in": 5,
        "tokens_in_total": 10,
        "cache_read_tokens": 3,
        "cache_write_tokens": 2,
        "tokens_out": 2,
    }

    assert cost(summary, 10, 50, 1, 12.5) == 0.000178
