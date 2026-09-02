from __future__ import annotations

from pathlib import Path

from run_fixed_schedule import (
    EPOCHS,
    FINAL_ONLY_EVAL_EVERY,
    final_epoch_config,
    recursive_sha256_manifest,
)


def test_final_epoch_config_disables_periodic_benchmark_selection():
    original = {"epochs": 50, "eval_every": 10, "lr": 1e-3}
    configured = final_epoch_config(original)

    assert original == {"epochs": 50, "eval_every": 10, "lr": 1e-3}
    assert configured["epochs"] == EPOCHS == 1000
    assert configured["eval_every"] == FINAL_ONLY_EVAL_EVERY == EPOCHS + 1
    assert configured["lr"] == original["lr"]


def test_recursive_manifest_includes_nested_inputs(tmp_path: Path):
    (tmp_path / "top.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    nested = tmp_path / "extras"
    nested.mkdir()
    (nested / "mask.txt").write_text("1 0 1\n", encoding="utf-8")

    manifest = recursive_sha256_manifest(tmp_path)

    assert set(manifest) == {"top.csv", "extras/mask.txt"}
    assert all(len(digest) == 64 for digest in manifest.values())
