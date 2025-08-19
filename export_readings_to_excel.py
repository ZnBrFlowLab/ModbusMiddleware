#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
export_readings_to_excel.py
从 SQLite (readings 表) 导出为 Excel。
特性：
- 支持按点位 name_full 过滤（可多次指定）
- 支持 since/until 时间范围（UNIX 时间戳或 ISO8601 自动识别）
- 每个点位一张工作表，另有汇总 Sheet（Combined）
"""

import argparse, sqlite3, time
from pathlib import Path
from typing import List, Optional
import pandas as pd
from datetime import datetime

def _parse_time(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    s = s.strip()
    try:
        return float(s)  # 直接当 Unix 时间戳
    except:
        pass
    for fmt in [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d",
        "%Y%m%d%H%M%S",
        "%Y%m%d",
    ]:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.timestamp()
        except:
            continue
    raise ValueError(f"无法解析时间: {s}")

def _read_names(conn) -> List[str]:
    cur = conn.execute("SELECT DISTINCT name FROM readings")
    return [r[0] for r in cur.fetchall()]

def _query(conn, name: str, since: Optional[float], until: Optional[float], limit: Optional[int]):
    sql = "SELECT ts, value FROM readings WHERE name=?"
    params: List = [name]
    if since is not None:
        sql += " AND ts>=?"
        params.append(float(since))
    if until is not None:
        sql += " AND ts<=?"
        params.append(float(until))
    sql += " ORDER BY ts ASC"
    if limit is not None and limit > 0:
        sql += f" LIMIT {int(limit)}"
    cur = conn.execute(sql, params)
    rows = cur.fetchall()
    return rows

def main():
    ap = argparse.ArgumentParser(description="导出 SQLite(readings) 为 Excel")
    ap.add_argument("--db", default="data.sqlite3", help="SQLite 路径（默认 data.sqlite3）")
    ap.add_argument("--out", default=None, help="输出 Excel 路径（默认 export_YYYYmmdd_HHMMSS.xlsx）")
    ap.add_argument("--name", action="append", help="指定点位 full_key（可多次指定）")
    ap.add_argument("--since", default=None, help="开始时间（UNIX 秒或常见日期格式，如 2025-08-01 00:00:00）")
    ap.add_argument("--until", default=None, help="结束时间（同上）")
    ap.add_argument("--limit", type=int, default=None, help="每个点位最多导出行数（默认不限）")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise FileNotFoundError(f"数据库不存在：{db_path}")

    since_ts = _parse_time(args.since) if args.since else None
    until_ts = _parse_time(args.until) if args.until else None

    out_path = args.out or f"export_{time.strftime('%Y%m%d_%H%M%S')}.xlsx"

    conn = sqlite3.connect(str(db_path))
    try:
        names = args.name or _read_names(conn)
        if not names:
            print("没有可导出的点位（readings 表为空）")
            return

        with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
            combined_frames = []
            for nm in names:
                rows = _query(conn, nm, since_ts, until_ts, args.limit)
                if not rows:
                    df_empty = pd.DataFrame(columns=["ts", "iso_time", "value"])
                    df_empty.to_excel(writer, sheet_name=(nm[:31] or "empty"), index=False)
                    continue
                df = pd.DataFrame(rows, columns=["ts", "value"])
                df["iso_time"] = df["ts"].apply(lambda x: datetime.utcfromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S"))
                df = df[["ts", "iso_time", "value"]]
                sheet = nm[:31] or "sheet"
                df.to_excel(writer, sheet_name=sheet, index=False)

                ws = writer.sheets[sheet]
                for i, col in enumerate(df.columns):
                    width = max(10, min(40, int(df[col].astype(str).str.len().max() if not df.empty else 10)))
                    ws.set_column(i, i, width)

                stats = {
                    "name": nm,
                    "rows": len(df),
                    "ts_min": float(df["ts"].min()),
                    "ts_max": float(df["ts"].max()),
                    "value_min": float(df["value"].min()) if df["value"].notna().any() else None,
                    "value_max": float(df["value"].max()) if df["value"].notna().any() else None,
                }
                combined_frames.append(pd.DataFrame([stats]))

            if combined_frames:
                summary = pd.concat(combined_frames, ignore_index=True)
                summary["iso_min"] = summary["ts_min"].apply(lambda x: datetime.utcfromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S") if pd.notna(x) else "")
                summary["iso_max"] = summary["ts_max"].apply(lambda x: datetime.utcfromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S") if pd.notna(x) else "")
                cols = ["name","rows","ts_min","iso_min","ts_max","iso_max","value_min","value_max"]
                summary = summary[cols]
                summary.to_excel(writer, sheet_name="Combined", index=False)
                ws = writer.sheets["Combined"]
                for i, col in enumerate(summary.columns):
                    width = max(10, min(40, int(summary[col].astype(str).str.len().max())))
                    ws.set_column(i, i, width)

        print(f"导出完成：{out_path}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
