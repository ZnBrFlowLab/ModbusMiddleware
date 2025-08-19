#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
export_readings_to_excel.py
按 (C,M,S) 分组导出为多个工作表：
- 每个工作表 = 一个实体（C{c}M{m}S{s}）
- 行：时间戳（ts, iso_time）
- 列：点位基础名（name）
- 解析失败或不含 C/M/S 的点位，写入 'Others' 表，逐条记录

可选过滤：
--name  可多次指定
--since/--until  时间范围（UNIX 或常见日期格式）
--limit  每个点位最多导出行数（先筛行，再合并）
"""

import argparse, sqlite3, time, re
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import pandas as pd
from datetime import datetime

# === 与中台一致的 full_key 解析（默认 {name}@C{c}M{m}S{s}） ===
_DEFAULT_FMT = "{name}@C{c}M{m}S{s}"
# 简单正则：末尾必须含 C\d+M\d+S\d+
_KEY_RE = re.compile(r"^(?P<base>.+?)@C(?P<c>-?\d+)M(?P<m>-?\d+)S(?P<s>-?\d+)$")

def parse_full_key(full_key: str) -> Tuple[str, Optional[int], Optional[int], Optional[int]]:
    m = _KEY_RE.match(full_key)
    if not m:
        return full_key, None, None, None
    base = m.group("base")
    c = int(m.group("c"))
    m_ = int(m.group("m"))
    s = int(m.group("s"))
    return base, c, m_, s

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

def _safe_sheet_name(s: str) -> str:
    # Excel sheet 名最长 31 且不含特殊字符
    s = re.sub(r"[:\\/?*\[\]]", "_", s)[:31]
    return s or "sheet"

def main():
    ap = argparse.ArgumentParser(description="按 (C,M,S) 分表导出 SQLite(readings) 为 Excel")
    ap.add_argument("--db", default="data.sqlite3", help="SQLite 路径（默认 data.sqlite3）")
    ap.add_argument("--out", default=None, help="输出 Excel 路径（默认 export_YYYYmmdd_HHMMSS.xlsx）")
    ap.add_argument("--name", action="append", help="仅导出指定 full_key（可多次指定）")
    ap.add_argument("--since", default=None, help="开始时间（UNIX 或 'YYYY-mm-dd HH:MM:SS' 等）")
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
        all_names = args.name or _read_names(conn)
        if not all_names:
            print("没有可导出的点位（readings 表为空）")
            return

        # 1) 先按 (C,M,S) 分桶；解析失败的进 Others
        groups: Dict[Tuple[int,int,int], List[str]] = {}
        others: List[str] = []

        base_name_map: Dict[str, str] = {}   # full_key -> base name
        cms_map: Dict[str, Tuple[int,int,int]] = {}  # full_key -> (c,m,s)

        for full in all_names:
            base, c, m, s = parse_full_key(full)
            base_name_map[full] = base
            if c is None or m is None or s is None:
                others.append(full)
            else:
                groups.setdefault((c, m, s), []).append(full)
                cms_map[full] = (c, m, s)

        # 2) 写 Excel：每个 (c,m,s) 一个工作表
        with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
            # 索引页（可选）
            index_rows = []

            for (c, m, s), name_list in sorted(groups.items()):
                # 逐点位查询并 merge 成宽表：行=ts，列=base name
                merged: Optional[pd.DataFrame] = None
                count_rows = 0

                for full in name_list:
                    rows = _query(conn, full, since_ts, until_ts, args.limit)
                    if not rows:
                        continue
                    df = pd.DataFrame(rows, columns=["ts", "value"])
                    df["iso_time"] = df["ts"].apply(lambda x: datetime.utcfromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S"))
                    col = base_name_map[full]  # 列名用基础名
                    df = df[["ts", "iso_time", "value"]].rename(columns={"value": col})

                    if merged is None:
                        merged = df
                    else:
                        # 以 ts 对齐，保留所有时间戳
                        merged = pd.merge(merged, df, on=["ts", "iso_time"], how="outer")

                    count_rows += len(df)

                sheet_name = _safe_sheet_name(f"C{c}M{m}S{s}")
                if merged is None or merged.empty:
                    # 写空表头
                    pd.DataFrame(columns=["ts","iso_time"]).to_excel(writer, sheet_name=sheet_name, index=False)
                    ws = writer.sheets[sheet_name]
                    ws.set_column(0, 1, 20)
                    index_rows.append({"sheet": sheet_name, "c": c, "m": m, "s": s, "rows": 0, "points": len(name_list)})
                    continue

                # 按时间排序
                merged = merged.sort_values("ts")
                # 提前设置列顺序：ts, iso_time, 再按名称排序
                value_cols = sorted([c for c in merged.columns if c not in ("ts","iso_time")])
                merged = merged[["ts","iso_time"] + value_cols]

                merged.to_excel(writer, sheet_name=sheet_name, index=False)
                ws = writer.sheets[sheet_name]
                # 列宽：时间列更宽，值列自适应（最大 40）
                ws.set_column(0, 0, 14)  # ts
                ws.set_column(1, 1, 20)  # iso_time
                for i, col in enumerate(value_cols, start=2):
                    width = max(10, min(40, int(merged[col].astype(str).str.len().max() if not merged.empty else 10)))
                    ws.set_column(i, i, width)

                index_rows.append({"sheet": sheet_name, "c": c, "m": m, "s": s, "rows": int(len(merged)), "points": len(value_cols)})

            # 3) Others：解析失败/无 C/M/S 的点位，逐条记录
            if others:
                frames = []
                for full in others:
                    rows = _query(conn, full, since_ts, until_ts, args.limit)
                    if not rows:
                        continue
                    df = pd.DataFrame(rows, columns=["ts", "value"])
                    df["iso_time"] = df["ts"].apply(lambda x: datetime.utcfromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S"))
                    df["name_full"] = full
                    df["base_name"] = base_name_map.get(full, full)
                    df = df[["ts","iso_time","name_full","base_name","value"]]
                    frames.append(df)
                if frames:
                    df_o = pd.concat(frames, ignore_index=True).sort_values("ts")
                    sheet_name = "Others"
                    df_o.to_excel(writer, sheet_name=sheet_name, index=False)
                    ws = writer.sheets[sheet_name]
                    for i, col in enumerate(df_o.columns):
                        width = max(10, min(40, int(df_o[col].astype(str).str.len().max())))
                        ws.set_column(i, i, width)
                    index_rows.append({"sheet": sheet_name, "c": "", "m": "", "s": "", "rows": int(len(df_o)), "points": int(df_o["name_full"].nunique())})

            # 4) 目录索引表
            if index_rows:
                df_idx = pd.DataFrame(index_rows)
                df_idx = df_idx[["sheet","c","m","s","rows","points"]]
                df_idx.to_excel(writer, sheet_name="Index", index=False)
                ws = writer.sheets["Index"]
                for i, col in enumerate(df_idx.columns):
                    width = max(8, min(30, int(df_idx[col].astype(str).str.len().max())))
                    ws.set_column(i, i, width)

        print(f"导出完成：{out_path}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()

