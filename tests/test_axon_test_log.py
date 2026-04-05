from __future__ import annotations

import os
from pathlib import Path

from brainsurgery.synapse.axon_test_log import render_axon_test_log


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_render_axon_test_log_uses_latest_parent_run(tmp_path: Path) -> None:
    old_worker = tmp_path / "log-100-a-b.txt"
    new_worker = tmp_path / "log-200-a-b.txt"
    _write(
        old_worker,
        "\n".join(
            [
                "result.axon=a.axon",
                "result.model_dir=/tmp/b",
                "result.masked_top1_eq=True",
                "result.masked_max_abs_diff=1.0",
                "result.masked_max_rel_diff=2.0",
            ]
        ),
    )
    _write(
        new_worker,
        "\n".join(
            [
                "result.axon=a.axon",
                "result.model_dir=/tmp/b",
                "result.masked_top1_eq=True",
                "result.masked_max_abs_diff=0.5",
                "result.masked_max_rel_diff=1.5",
            ]
        ),
    )
    old_parent = tmp_path / "parent-100.txt"
    new_parent = tmp_path / "parent-200.txt"
    _write(old_parent, f"child_start pair_index=0 pid=100 log_path={old_worker}\n")
    _write(new_parent, f"child_start pair_index=0 pid=200 log_path={new_worker}\n")
    os.utime(old_parent, (1000, 1000))
    os.utime(new_parent, (2000, 2000))

    rendered = render_axon_test_log(tmp_path)

    assert "`0.5`" in rendered
    assert "`1.0`" not in rendered


def test_render_axon_test_log_all_deduplicates_by_latest_pair(tmp_path: Path) -> None:
    worker_a_old = tmp_path / "log-100-a-x.txt"
    worker_a_new = tmp_path / "log-200-a-x.txt"
    worker_b = tmp_path / "log-201-b-y.txt"
    _write(
        worker_a_old,
        "\n".join(
            [
                "result.axon=a.axon",
                "result.model_dir=/tmp/x",
                "result.masked_top1_eq=False",
                "result.masked_max_abs_diff=9.0",
                "result.masked_max_rel_diff=9.0",
            ]
        ),
    )
    _write(
        worker_a_new,
        "\n".join(
            [
                "result.axon=a.axon",
                "result.model_dir=/tmp/x",
                "result.masked_top1_eq=True",
                "result.masked_max_abs_diff=0.25",
                "result.masked_max_rel_diff=0.5",
            ]
        ),
    )
    _write(
        worker_b,
        "\n".join(
            [
                "result.axon=b.axon",
                "result.model_dir=/tmp/y",
                "result.masked_top1_eq=True",
                "result.masked_max_abs_diff=0.75",
                "result.masked_max_rel_diff=0.9",
            ]
        ),
    )
    _write(
        tmp_path / "parent-100.txt", f"child_start pair_index=0 pid=100 log_path={worker_a_old}\n"
    )
    _write(
        tmp_path / "parent-200.txt",
        "\n".join(
            [
                f"child_start pair_index=0 pid=200 log_path={worker_a_new}",
                f"child_start pair_index=1 pid=201 log_path={worker_b}",
            ]
        ),
    )
    os.utime(tmp_path / "parent-100.txt", (1000, 1000))
    os.utime(tmp_path / "parent-200.txt", (2000, 2000))

    rendered = render_axon_test_log(tmp_path, all_runs=True)

    assert rendered.count("| `a.axon` | `x` |") == 1
    assert "`0.25`" in rendered
    assert "`9.0`" not in rendered
    assert "| `b.axon` | `y` | `True` | `0.75` | `0.9` |" in rendered
