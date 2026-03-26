"""Copy scripts/configs and write config_backup.txt for reproducibility."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def copy_artifacts(
    exp_dir: Path,
    main_script: Path,
    extra_files: list[Path],
) -> None:
    art = exp_dir / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    if main_script.is_file():
        shutil.copy2(main_script, art / main_script.name)
    for p in extra_files:
        if p.is_file():
            shutil.copy2(p, art / p.name)
        elif p.is_dir():
            shutil.copytree(p, art / p.name)


def write_config_backup(exp_dir: Path, cfg: dict[str, Any], cli_overrides: dict[str, Any]) -> None:
    """Persist cfg after caller-applied CLI merges (e.g. db_table, output_root) plus raw cli_overrides."""
    lines = [
        "# merged config (yaml-loaded dict) + CLI overrides",
        "",
        "## full_config (json)",
        json.dumps(cfg, indent=2, ensure_ascii=False, default=str),
        "",
        "## cli_overrides",
        json.dumps(cli_overrides, indent=2, ensure_ascii=False, default=str),
    ]
    (exp_dir / "config_backup.txt").write_text("\n".join(lines), encoding="utf-8")
