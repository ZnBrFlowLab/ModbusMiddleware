#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re, math, json
from pathlib import Path
import pandas as pd

SHEET_LEVEL = {"组串":"cluster", "模块":"module", "子堆":"substack"}

def parse_digits_from_text(s: str|None):
    if not isinstance(s, str): return None
    m = re.search(r"保留小数点后(\d+)位", s)
    if m:
        return int(m.group(1))
    return None

def rw_from_text(s: str|None):
    if not isinstance(s, str): return "R"
    s = s.strip()
    if "只读" in s: return "R"
    if "可读可写" in s or "读写" in s: return "RW"
    if "只写" in s: return "W"
    return "R"

def infer_type(t: str|None):
    if not isinstance(t, str): return "uint16"
    t = t.strip().lower()
    if "int32" in t: return "int32"
    if "uint32" in t: return "uint32"
    if "float" in t: return "float32"
    return "uint16"

def main(xlsx_path: str, out_path="generated_tagmap.json"):
    xls = pd.read_excel(xlsx_path, sheet_name=None)
    points = []
    for sheet, df in xls.items():
        if sheet not in SHEET_LEVEL: 
            continue
        level = SHEET_LEVEL[sheet]
        cols = list(df.columns)
        addr_col = None
        low_bits = None
        for c in cols:
            s = str(c)
            if "地址（低" in s and "位" in s:
                addr_col = c
                m = re.search(r"低(\d+)位", s)
                if m:
                    low_bits = int(m.group(1))
                break
        if addr_col is None:
            continue
        for _, r in df.iterrows():
            name = str(r.get("内容","")).strip()
            if not name or name == "nan":
                continue
            try:
                raw = r.get(addr_col)
                if isinstance(raw, str) and raw.strip().lower().startswith('0x'):
                    low_addr = int(raw, 16)
                else:
                    low_addr = int(raw)
            except Exception:
                continue
            typ = infer_type(str(r.get("类型","")))
            rw  = rw_from_text(str(r.get("状态","")))
            desc= str(r.get("说明","") or "").strip()
            digits = parse_digits_from_text(desc)
            scale = math.pow(10, -digits) if digits is not None else None

            points.append({
                "name": name,
                "slave": 1,
                "func": 3 if rw=="R" else (6 if rw=="W" else 3),
                "addr": int(low_addr),
                "type": typ,
                "rw": rw,
                "scale": scale,
                "desc": desc,
                "level": level,
                "low_bits": low_bits,
            })
    out = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "addr_composition": "runtime",
        "points": points,
    }
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已生成: {out_path}（{len(points)} 点）")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("excel")
    ap.add_argument("--out", default="generated_tagmap.json")
    args = ap.parse_args()
    main(args.excel, args.out)
