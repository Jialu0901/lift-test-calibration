"""
MySQL 连接配置：仅从 JSON 读取（不依赖 src2）。

JSON 路径优先级：
1. 环境变量 LIFT_DB_CONFIG_JSON（绝对或相对路径，相对路径按当前工作目录解析）
2. 默认：``feature_engineering/db_config.json``（相对本文件为
   ``Path(__file__).resolve().parent.parent.parent / "db_config.json"``）

读取字段：host, user, password, database, port；其余键忽略。
port 缺省为 3306。首次调用 ``get_db_config()`` 或访问 ``DB_CONFIG`` 时加载并缓存。

宽表名不再由此模块提供，须由调用方传入；``DEFAULT_WIDE_TABLE`` 恒为 ``None``。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_WIDE_TABLE: str | None = None

_config_cache: dict[str, Any] | None = None


def _default_json_path() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "db_config.json"


def resolve_db_config_json_path() -> Path:
    """Resolved path to the MySQL config JSON (for logging / diagnostics)."""
    env = os.environ.get("LIFT_DB_CONFIG_JSON")
    if env:
        p = Path(os.path.expanduser(env.strip()))
        return p.resolve()
    return _default_json_path()


def _load_from_json() -> dict[str, Any]:
    path = resolve_db_config_json_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"MySQL config JSON not found: {path}. "
            "Set LIFT_DB_CONFIG_JSON or place db_config.json under feature_engineering/."
        )
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"DB config JSON must be a JSON object: {path}")
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


def get_db_config() -> dict[str, Any]:
    """Load and cache DB connection dict from JSON (lazy)."""
    global _config_cache
    if _config_cache is None:
        _config_cache = _load_from_json()
    return _config_cache


def __getattr__(name: str) -> Any:
    if name == "DB_CONFIG":
        return get_db_config()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
