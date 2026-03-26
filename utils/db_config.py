"""
MySQL 连接配置：JSON 文件或 Notebook 内存（``set_db_config_inline``），不依赖 src2。

**Notebook 推荐同一变量名** ``DB_CONFIG``：

- 从文件：``DB_CONFIG = read_db_config_json()`` 或 ``read_db_config_json(Path(...))``
- 手写：``DB_CONFIG = {"host": ..., "user": ..., ...}``

再 ``run_pipeline_notebook(..., db_config=DB_CONFIG)``，或省略字典、直接
``run_pipeline_notebook(..., db_config_json_path=Path(".../db_config.json"))``；亦可 ``set_db_config_inline(DB_CONFIG)``。

``get_db_config()`` 优先级：

1. **内存**：已 ``set_db_config_inline`` 或流水线传入的 inline
2. 环境变量 ``LIFT_DB_CONFIG_JSON`` 指向的文件
3. 默认：仓库根 ``db_config.json`` 或 monorepo ``feature_engineering/db_config.json``

读取字段：host, user, password, database, port；其余键忽略。port 缺省 3306。

宽表名不由本模块提供；``DEFAULT_WIDE_TABLE`` 恒为 ``None``。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_WIDE_TABLE: str | None = None

_config_cache: dict[str, Any] | None = None
_inline_normalized: dict[str, Any] | None = None


def _default_json_path() -> Path:
    """GitHub 扁平仓库：db_config.json 在仓库根；monorepo：在 feature_engineering/。"""
    utils_dir = Path(__file__).resolve().parent
    mb_or_repo_root = utils_dir.parent
    feature_engineering = mb_or_repo_root.parent
    at_repo = mb_or_repo_root / "db_config.json"
    at_fe = feature_engineering / "db_config.json"
    if at_repo.is_file():
        return at_repo
    if at_fe.is_file():
        return at_fe
    return at_repo


def resolve_db_config_json_path() -> Path:
    """Resolved path to the MySQL config JSON (for logging / diagnostics)."""
    if _inline_normalized is not None:
        return Path("<DB_CONFIG_INLINE>")
    env = os.environ.get("LIFT_DB_CONFIG_JSON")
    if env:
        p = Path(os.path.expanduser(env.strip()))
        return p.resolve()
    return _default_json_path()


def read_db_config_json(path: str | Path | None = None) -> dict[str, Any]:
    """
    Read MySQL settings from a JSON file into a dict (same keys as inline: host, user, password, database, port).

    Does **not** set global inline state; use with ``run_pipeline_notebook(..., db_config=...)`` or
    ``set_db_config_inline(DB_CONFIG)``.

    - ``path`` given: read that file.
    - ``path`` is ``None``: ``LIFT_DB_CONFIG_JSON`` if set, else ``_default_json_path()`` (first existing
      repo-root or feature_engineering ``db_config.json``).
    """
    if path is not None:
        p = Path(path).expanduser().resolve()
    else:
        env = os.environ.get("LIFT_DB_CONFIG_JSON")
        if env:
            p = Path(os.path.expanduser(env.strip())).resolve()
        else:
            p = _default_json_path()
    if not p.is_file():
        raise FileNotFoundError(
            f"MySQL config JSON not found: {p}. "
            "Pass a path to read_db_config_json, set LIFT_DB_CONFIG_JSON, place db_config.json at repo root, "
            "or define a dict and use set_db_config_inline / run_pipeline_notebook(db_config=...)."
        )
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"DB config JSON must be a JSON object: {p}")
    return _normalize_db_mapping(raw)


def _normalize_db_mapping(raw: dict[str, Any]) -> dict[str, Any]:
    host = str(raw.get("host") or "").strip()
    user = str(raw.get("user") or "").strip()
    password = str(raw.get("password") or "")
    database = str(raw.get("database") or "").strip()
    port_raw = raw.get("port", 3306)
    port = int(port_raw) if port_raw is not None else 3306
    return {
        "host": host,
        "user": user,
        "password": password,
        "database": database,
        "port": port,
    }


def set_db_config_inline(raw: dict[str, Any] | None) -> None:
    """
    Use in-memory DB credentials (e.g. notebook globals). Pass None to clear and fall back to JSON file.

    Call before any code that invokes get_db_config(). Clears file-based cache.
    """
    global _inline_normalized, _config_cache
    _config_cache = None
    _inline_normalized = _normalize_db_mapping(raw) if raw is not None else None


def _load_from_json() -> dict[str, Any]:
    path = resolve_db_config_json_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"MySQL config JSON not found: {path}. "
            "Use set_db_config_inline({...}) in a notebook, set LIFT_DB_CONFIG_JSON, "
            "or copy db_config.example.json to db_config.json at repo root "
            "(or feature_engineering/db_config.json in a monorepo)."
        )
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"DB config JSON must be a JSON object: {path}")
    return _normalize_db_mapping(raw)


def get_db_config() -> dict[str, Any]:
    """Load and cache DB connection dict (inline, or from JSON on first use)."""
    global _config_cache
    if _inline_normalized is not None:
        return _inline_normalized
    if _config_cache is None:
        _config_cache = _load_from_json()
    return _config_cache


def __getattr__(name: str) -> Any:
    if name == "DB_CONFIG":
        return get_db_config()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
