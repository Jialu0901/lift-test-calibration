"""
从 MySQL 读取建模宽表。连接信息见 ``utils.db_config``（JSON）；表名必须由调用方传入，无默认。
"""
from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import create_engine, text

from utils.db_config import get_db_config

logger = logging.getLogger(__name__)


def _build_engine() -> Any:
    cfg = dict(get_db_config())
    user = cfg.pop("user")
    password = cfg.pop("password")
    host = cfg.pop("host")
    port = cfg.pop("port", 3306)
    database = cfg.pop("database", "workmagic_bi")
    pass_enc = quote_plus(str(password))
    url = f"mysql+mysqlconnector://{user}:{pass_enc}@{host}:{port}/{database}"
    return create_engine(url, pool_pre_ping=True)


def _set_timeouts(conn: Any) -> None:
    try:
        conn.execute(text("SET SESSION net_read_timeout = 3600"))
        conn.execute(text("SET SESSION net_write_timeout = 3600"))
        conn.commit()
    except Exception:
        pass


def load_wide_table_from_db(
    table_name: str | None = None,
    limit: int | None = None,
    date_range: tuple[str, str] | None = None,
    outcome_date_col: str = "outcome_date",
    chunk_days: int | None = 1,
) -> pd.DataFrame:
    """
    SELECT * FROM 宽表。全表或显式 date_range 时均可按 outcome_date 分块拉取，降低超时 / 2013 断连风险。

    Args:
        table_name: 必填，非空字符串（``schema.table`` 或当前库下的表名）
        limit: 调试用 LIMIT（与 date_range 并存时不走分块）
        date_range: (start, end) YYYY-MM-DD
        chunk_days: 每个查询覆盖的**连续日历天数**（含起止日）。单日数据量很大时应用 **1**
            （每个 outcome_date 一次查询）。>0 且提供 date_range、无 limit 时按窗口多次查询；
            None 或 <=0 时对 date_range 单次拉全区间（易超大结果集失败）
    """
    final_table = (table_name or "").strip()
    if not final_table:
        raise ValueError(
            "table_name is required and must be non-empty when loading from the database; "
            "pass db_table (e.g. YAML or --db-table) or use wide_table_path for parquet."
        )
    table_name = final_table
    engine = _build_engine()

    # Split date_range into windows (run_pipeline always passes date_range from splits)
    if (
        date_range is not None
        and limit is None
        and chunk_days is not None
        and chunk_days > 0
    ):
        d0 = pd.Timestamp(date_range[0]).date()
        d1 = pd.Timestamp(date_range[1]).date()
        chunks: list[pd.DataFrame] = []
        current = d0
        chunk_num = 0
        while current <= d1:
            # chunk_days = inclusive span (1 => single outcome_date per query)
            end_d = min(current + timedelta(days=chunk_days - 1), d1)
            q = f"SELECT * FROM {table_name} WHERE {outcome_date_col} BETWEEN :start AND :end"
            # Fresh connection per chunk so a dropped connection on one query does not poison the rest
            with engine.connect() as conn:
                _set_timeouts(conn)
                chunk_df = pd.read_sql_query(
                    text(q), conn, params={"start": str(current), "end": str(end_d)}
                )
            if not chunk_df.empty:
                chunks.append(chunk_df)
                chunk_num += 1
                logger.info(
                    "Loaded chunk %d: %s to %s (%d rows)",
                    chunk_num,
                    current,
                    end_d,
                    len(chunk_df),
                )
            current = end_d + timedelta(days=1)
        if not chunks:
            return pd.DataFrame()
        return pd.concat(chunks, ignore_index=True)

    if limit is not None or date_range is not None:
        q = f"SELECT * FROM {table_name}"
        params: dict[str, Any] = {}
        if date_range is not None:
            q += f" WHERE {outcome_date_col} BETWEEN :date_start AND :date_end"
            params["date_start"] = date_range[0]
            params["date_end"] = date_range[1]
        if limit is not None:
            q += " LIMIT :limit_val"
            params["limit_val"] = limit
        with engine.connect() as conn:
            _set_timeouts(conn)
            if params:
                return pd.read_sql_query(text(q), conn, params=params)
            return pd.read_sql_query(text(q), conn)

    with engine.connect() as conn:
        _set_timeouts(conn)
        range_q = text(
            f"SELECT MIN({outcome_date_col}) as min_d, MAX({outcome_date_col}) as max_d FROM {table_name}"
        )
        try:
            range_df = pd.read_sql_query(range_q, conn)
        except Exception:
            range_df = pd.DataFrame()
        if range_df.empty or range_df["min_d"].iloc[0] is None or range_df["max_d"].iloc[0] is None:
            return pd.DataFrame()
        min_d = pd.Timestamp(range_df["min_d"].iloc[0]).date()
        max_d = pd.Timestamp(range_df["max_d"].iloc[0]).date()
        if chunk_days is None or chunk_days <= 0:
            q = f"SELECT * FROM {table_name}"
            return pd.read_sql_query(text(q), conn)
        chunks: list[pd.DataFrame] = []
        current = min_d
        chunk_num = 0
        while current <= max_d:
            end_d = min(current + timedelta(days=chunk_days - 1), max_d)
            q = f"SELECT * FROM {table_name} WHERE {outcome_date_col} BETWEEN :start AND :end"
            with engine.connect() as chunk_conn:
                _set_timeouts(chunk_conn)
                chunk_df = pd.read_sql_query(
                    text(q),
                    chunk_conn,
                    params={"start": str(current), "end": str(end_d)},
                )
            if not chunk_df.empty:
                chunks.append(chunk_df)
                chunk_num += 1
                logger.info(
                    "Loaded chunk %d: %s to %s (%d rows)",
                    chunk_num,
                    current,
                    end_d,
                    len(chunk_df),
                )
            current = end_d + timedelta(days=1)
        if not chunks:
            return pd.DataFrame()
        return pd.concat(chunks, ignore_index=True)
