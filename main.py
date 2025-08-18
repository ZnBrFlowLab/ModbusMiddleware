#!/usr/bin/env python
# -*- coding: utf-8 -*-
import asyncio, json, yaml, time, math, sqlite3, logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from pydantic import BaseModel
from pymodbus.client import ModbusSerialClient, ModbusTcpClient
from pymodbus.payload import BinaryPayloadBuilder, BinaryPayloadDecoder
from pymodbus.constants import Endian

# ---------------- 基本配置 ---------------- #
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
app = FastAPI(title="Modbus 中台", version="1.2.0")

CONFIG_PATH = Path(__file__).with_name("config.yaml")
TAGMAP_PATH = None
CFG = None
TAGMAP = None

DB_PATH = Path(__file__).with_name("data.sqlite3")
DB_Q: "asyncio.Queue[Tuple[str,float,Any,bytes]]" = asyncio.Queue(maxsize=10000)

# 内存缓存：{full_key: {"value": v, "ts": t, "raw": raw_ints} }
CACHE: Dict[str, Dict[str, Any]] = {}

# WS 连接管理
class ConnManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        try:
            snapshot = {k:v["value"] for k,v in CACHE.items()}
            meta = build_meta_dict_for_keys(list(CACHE.keys())) if CFG.get("include_meta", True) else None
            payload = {"type":"snapshot","data":snapshot}
            if meta is not None:
                payload["meta"] = meta
            await ws.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception:
            pass

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: Dict[str, Any]):
        text = json.dumps(data, ensure_ascii=False)
        for ws in list(self.active):
            try:
                await ws.send_text(text)
            except Exception:
                self.disconnect(ws)

ws_manager = ConnManager()

# ---------------- 加载配置/点表 ---------------- #
def load_config():
    global CFG, TAGMAP_PATH
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        CFG = yaml.safe_load(f)
    TAGMAP_PATH = Path(CFG.get("tagmap_path", "generated_tagmap.json")).resolve()

def load_tagmap():
    global TAGMAP
    if not TAGMAP_PATH.exists():
        raise FileNotFoundError(f"点表不存在: {TAGMAP_PATH}")
    TAGMAP = json.loads(TAGMAP_PATH.read_text(encoding="utf-8"))

# ---------------- Modbus 客户端（同步） ---------------- #
client = None

def make_client():
    mode = str(CFG.get("mode","rtu")).lower()
    if mode == "rtu":
        c = CFG["rtu"]
        return ModbusSerialClient(
            method="rtu",
            port=c["port"],
            baudrate=int(c["baudrate"]),
            parity=c.get("parity","N"),
            stopbits=int(c.get("stopbits",1)),
            bytesize=int(c.get("bytesize",8)),
            timeout=float(c.get("timeout",1.0)),
        )
    elif mode == "tcp":
        c = CFG["tcp"]
        return ModbusTcpClient(c["host"], port=int(c["port"]), timeout=float(c.get("timeout",1.0)))
    else:
        raise ValueError("mode 必须是 rtu 或 tcp")

# ---------------- 工具：范围解析/键名/元信息 ---------------- #
def _parse_range(spec, fallback: List[int]) -> List[int]:
    """支持：int / list / 'a-b' 字符串。如果 spec 为空则用 fallback。"""
    if spec is None:
        return fallback
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, list):
        return [int(x) for x in spec]
    if isinstance(spec, str) and "-" in spec:
        a, b = spec.split("-", 1)
        a, b = int(a.strip()), int(b.strip())
        if a <= b:
            return list(range(a, b+1))
        else:
            return list(range(a, b-1, -1))
    return [int(spec)]

def id_combinations_for_level(level: str) -> List[Tuple[int,int,int]]:
    """按点位层级展开 (c,m,s) 列表。不存在的维度用 -1 占位。"""
    ranges = CFG.get("ranges", {}) or {}
    clusters = _parse_range(ranges.get("clusters"), [int(CFG.get("cluster_id",0))])
    modules  = _parse_range(ranges.get("modules"),  [int(CFG.get("module_id",0))])
    subs     = _parse_range(ranges.get("subs"),     [int(CFG.get("sub_id",0))])

    out = []
    if level == "cluster":
        for c in clusters:
            out.append((c, -1, -1))
    elif level == "module":
        for c in clusters:
            for m in modules:
                out.append((c, m, -1))
    else:  # substack
        for c in clusters:
            for m in modules:
                for s in subs:
                    out.append((c, m, s))
    return out

def build_full_key(name: str, c: int, m: int, s: int) -> str:
    fmt = CFG.get("key_format", "{name}@C{c}M{m}S{s}")
    return fmt.format(name=name, c=c, m=m, s=s)

def parse_full_key(full_key: str) -> Tuple[str,int,int,int]:
    """
    尝试从 full_key 反推 (name,c,m,s)。
    仅在无法从 meta 找到时用于写入等。
    约定：key_format 默认 {name}@C{c}M{m}S{s}，否则无法通用解析，需要前端带 ids。
    """
    default_fmt = "{name}@C{c}M{m}S{s}"
    if CFG.get("key_format", default_fmt) != default_fmt:
        # 不支持通用解析
        return (full_key, -1, -1, -1)
    if "@C" not in full_key or "M" not in full_key or "S" not in full_key:
        return (full_key, -1, -1, -1)
    try:
        name, rest = full_key.split("@C", 1)
        mpos = rest.rfind("M")
        spos = rest.rfind("S")
        c = int(rest[:mpos])
        m = int(rest[mpos+1:spos])
        s = int(rest[spos+1:])
        return (name, c, m, s)
    except Exception:
        return (full_key, -1, -1, -1)

def build_meta_dict_for_keys(keys: List[str]) -> Dict[str, Dict[str,int]]:
    meta = {}
    default_fmt = "{name}@C{c}M{m}S{s}"
    for k in keys:
        if CFG.get("key_format", default_fmt) == default_fmt:
            n,c,m,s = parse_full_key(k)
            meta[k] = {"c": c, "m": m, "s": s}
        else:
            # 若自定义了 key_format，这里仅返回空占位；建议前端依赖服务端返回的 meta（写入时也传 ids）
            meta[k] = {}
    return meta

# ---------------- 数据类型编解码 ---------------- #
def decode_value(typ: str, regs: List[int], scale: Optional[float]) -> Any:
    typ = (typ or "uint16").lower()
    value = None
    if typ in ("uint16","int16","bool"):
        raw = regs[0] if regs else 0
        if typ == "int16":
            if raw > 0x7FFF: raw -= 0x10000
            value = raw
        elif typ == "bool":
            value = bool(raw)
        else:
            value = raw
    elif typ in ("uint32","int32","float32"):
        if len(regs) < 2:
            return None
        decoder = BinaryPayloadDecoder.fromRegisters(regs, byteorder=Endian.BIG, wordorder=Endian.BIG)
        if typ == "float32":
            value = decoder.decode_32bit_float()
        elif typ == "int32":
            value = decoder.decode_32bit_int()
        else:
            value = decoder.decode_32bit_uint()
    else:
        raw = regs[0] if regs else 0
        value = raw

    if (scale is not None) and isinstance(value, (int, float)):
        s = (scale or 1.0)
        value = value * s
    return value

def encode_value(typ: str, value: Any, scale: Optional[float]) -> List[int]:
    typ = (typ or "uint16").lower()
    v = value
    if (scale is not None) and isinstance(v, (int,float)):
        s = (scale or 1.0)
        if s != 0:
            v = v / s

    regs: List[int] = []
    if typ == "bool":
        regs = [1 if v else 0]
    elif typ in ("uint16","int16"):
        iv = int(round(float(v)))
        if typ == "int16" and iv < 0:
            iv = (iv + (1<<16)) & 0xFFFF
        regs = [iv & 0xFFFF]
    elif typ in ("uint32","int32","float32"):
        builder = BinaryPayloadBuilder(byteorder=Endian.BIG, wordorder=Endian.BIG)
        if typ == "float32":
            builder.add_32bit_float(float(v))
        elif typ == "int32":
            builder.add_32bit_int(int(v))
        else:
            builder.add_32bit_uint(int(v))
        regs = builder.to_registers()
    else:
        regs = [int(round(float(v))) & 0xFFFF]
    return regs

def reg_len_of_type(typ: str) -> int:
    typ = (typ or "uint16").lower()
    return 2 if typ in ("uint32","int32","float32") else 1

# ---- 地址组合：使用传入的 (c,m,s) 构造 ----
def compose_addr_with_ids(pt: Dict[str, Any], c: int, m: int, s: int) -> int:
    layout = CFG.get("addr_layout", {})
    level = pt.get("level","cluster")
    low   = int(pt.get("addr", 0))
    if level == "module":
        if "module" in layout:
            base = int(layout["module"].get("base", 0))
            shifts = layout["module"].get("shifts", {"cluster":9, "module":6})
            return (int(c) << int(shifts["cluster"])) | (int(m) << int(shifts["module"])) | low | base
        return (int(c) << 9) | (int(m) << 6) | low
    elif level == "substack":
        if "substack" in layout:
            base = int(layout["substack"].get("base", 0))
            shifts = layout["substack"].get("shifts", {"cluster":10, "module":7, "sub":6})
            return (int(c) << int(shifts["cluster"])) | (int(m) << int(shifts["module"])) | (int(s) << int(shifts["sub"])) | low | base
        return (int(c) << 10) | (int(m) << 7) | (int(s) << 6) | low
    else:
        if "cluster" in layout:
            base = int(layout["cluster"].get("base", 0))
            shift = int(layout["cluster"].get("shift", 8))
            return (int(c) << shift) | low | base
        return (int(c) << 8) | low

# ---------------- 点位与“范围展开+分组” ---------------- #
def points() -> List[Dict[str, Any]]:
    return TAGMAP["points"]

def expanded_read_items(max_batch_len: int = 60) -> Dict[Tuple[int,int,int,int,int], List[Tuple[int,Dict[str,Any],Tuple[int,int,int]]]]:
    """
    将可读点位按 (slave, func, c, m, s) 分组，并按最终地址排序。
    返回: { (slave, func, c, m, s): [(addr, point, (c,m,s)), ...] }
    """
    groups: Dict[Tuple[int,int,int,int,int], List[Tuple[int,Dict[str,Any],Tuple[int,int,int]]]] = {}
    for p in points():
        if p.get("rw","R") not in ("R","RW"):
            continue
        level = p.get("level","cluster")
        combos = id_combinations_for_level(level)
        for (c,m,s) in combos:
            # 对于不存在的维度(-1)，compose 时不使用，但 key 里保留 -1 以标识层级
            cid = (c if c != -1 else int(CFG.get("cluster_id",0)))
            mid = (m if m != -1 else int(CFG.get("module_id",0)))
            sid = (s if s != -1 else int(CFG.get("sub_id",0)))
            addr = compose_addr_with_ids(p, cid, mid, sid)
            key  = (int(p["slave"]), int(p["func"]), c, m, s)
            groups.setdefault(key, []).append((addr, p, (cid, mid, sid)))
    # 排序
    for k in list(groups.keys()):
        groups[k].sort(key=lambda x: x[0])
    return groups

# ---------------- SQLite 持久化 ---------------- #
def init_db():
    conn = sqlite3.connect(DB_PATH)
    with conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS readings(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,   -- full_key
            ts   REAL NOT NULL,
            value REAL,
            raw  BLOB
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_readings_name_ts ON readings(name, ts)")
    conn.close()

def db_insert_many(rows: List[Tuple[str,float,Any,bytes]]):
    conn = sqlite3.connect(DB_PATH)
    with conn:
        conn.executemany("INSERT INTO readings(name, ts, value, raw) VALUES (?,?,?,?)", rows)
    conn.close()

async def db_writer_task():
    BATCH = 200
    INTERVAL = 1.0
    buf: List[Tuple[str,float,Any,bytes]] = []
    last_flush = time.time()
    while True:
        try:
            timeout = max(0.0, last_flush + INTERVAL - time.time())
            try:
                item = await asyncio.wait_for(DB_Q.get(), timeout=timeout)
                buf.append(item)
            except asyncio.TimeoutError:
                pass
            if len(buf) >= BATCH or (buf and time.time() - last_flush >= INTERVAL):
                await asyncio.to_thread(db_insert_many, buf[:])
                buf.clear()
                last_flush = time.time()
        except Exception as e:
            logging.exception("DB 写入失败: %s", e)
            await asyncio.sleep(1.0)

# ---------------- 轮询主循环（I/O 放线程） ---------------- #
_last_health = {"ok": False, "last_poll_ts": 0.0, "backoff": 0.5}

async def poll_loop():
    global client, _last_health
    poll_ms = int(CFG.get("poll_interval_ms", 500))
    max_batch_len = int(CFG.get("max_batch_len", 60))
    backoff = 0.5
    float_eps = float(CFG.get("float_epsilon", 1e-6))

    while True:
        try:
            if (client is None) or (not getattr(client, "connected", False)):
                client = make_client()
                await asyncio.to_thread(client.connect)

            groups = expanded_read_items(max_batch_len)
            updates = {}

            for (slave, func, c_mark, m_mark, s_mark), items in groups.items():
                i = 0
                while i < len(items):
                    start_addr = items[i][0]
                    length = 0
                    j = i
                    last_addr = start_addr - 1

                    while j < len(items):
                        addr, p, (cid, mid, sid) = items[j]
                        gap = addr - last_addr - 1
                        need = reg_len_of_type(p.get("type","uint16"))
                        new_len = (length + gap + need)
                        if new_len > max_batch_len:
                            break
                        length = new_len
                        last_addr = addr + need - 1
                        j += 1

                    if func in (3,4) and length > 0:
                        if func == 3:
                            rr = await asyncio.to_thread(client.read_holding_registers, start_addr, length, slave=slave)
                        else:
                            rr = await asyncio.to_thread(client.read_input_registers, start_addr, length, slave=slave)

                        if (not rr) or rr.isError():
                            logging.warning("读失败 slave=%s func=%s start=%s len=%s: %s", slave, func, start_addr, length, rr)
                            i = j
                            continue

                        regs_all = rr.registers
                        for k in range(i, j):
                            addr, p, (cid, mid, sid) = items[k]
                            typ = p.get("type","uint16")
                            need = reg_len_of_type(typ)
                            offset = addr - start_addr
                            if offset < 0 or offset + need > len(regs_all):
                                continue
                            chunk = regs_all[offset:offset+need]
                            val = decode_value(typ, chunk, p.get("scale"))
                            now = time.time()

                            full_key = build_full_key(p["name"], cid if c_mark!=-1 else -1, mid if m_mark!=-1 else -1, sid if s_mark!=-1 else -1)
                            prev = CACHE.get(full_key)
                            changed = (prev is None)
                            if isinstance(val, float) and prev and isinstance(prev.get("value"), float):
                                changed = (abs(val - prev["value"]) > float_eps)
                            elif prev:
                                changed = (val != prev.get("value"))

                            if changed:
                                CACHE[full_key] = {"value": val, "ts": now, "raw": chunk}
                                updates[full_key] = val
                                raw_bytes = b"".join(int(x).to_bytes(2, "big") for x in chunk)
                                try:
                                    DB_Q.put_nowait((full_key, now, float(val) if isinstance(val,(int,float)) else None, raw_bytes))
                                except asyncio.QueueFull:
                                    logging.warning("DB 队列已满，丢弃: %s", full_key)
                    i = j

            if updates:
                payload = {"type":"update","data":updates}
                if CFG.get("include_meta", True):
                    payload["meta"] = build_meta_dict_for_keys(list(updates.keys()))
                await ws_manager.broadcast(payload)

            _last_health.update({"ok": True, "last_poll_ts": time.time(), "backoff": 0.5})
            backoff = 0.5
            await asyncio.sleep(poll_ms/1000.0)

        except Exception as e:
            logging.exception("轮询异常: %s", e)
            _last_health.update({"ok": False, "last_poll_ts": time.time(), "backoff": backoff})
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 5.0)

# ---------------- API ---------------- #
class WriteReq(BaseModel):
    # 方式一：直接写 full_key（推荐前端存这个），例如 "pump_enable@C0M1S-1"
    name_full: Optional[str] = None
    # 方式二：提供基础名 + 指定 C/M/S（适合自定义 key_format 或希望结构化）
    name: Optional[str] = None
    cluster_id: Optional[int] = None
    module_id: Optional[int] = None
    sub_id: Optional[int] = None
    # 值
    value: float | int | bool

@app.on_event("startup")
async def _startup():
    load_config()
    load_tagmap()
    init_db()
    asyncio.create_task(poll_loop())
    asyncio.create_task(db_writer_task())
    asyncio.create_task(_heartbeat_task())

@app.on_event("shutdown")
async def _shutdown():
    try:
        if client and getattr(client, "connected", False):
            await asyncio.to_thread(client.close)
    except Exception:
        pass

@app.get("/tags")
def list_tags():
    return points()

@app.get("/values")
def all_values():
    data = {k:v["value"] for k,v in CACHE.items()}
    if CFG.get("include_meta", True):
        return {"data": data, "meta": build_meta_dict_for_keys(list(data.keys()))}
    return data

@app.get("/values/{name_full}")
def value_by_full_name(name_full: str):
    if name_full not in CACHE:
        raise HTTPException(404, f"{name_full} 暂无缓存或不存在")
    payload = CACHE[name_full]
    if CFG.get("include_meta", True):
        return {"data": payload, "meta": {name_full: build_meta_dict_for_keys([name_full])[name_full]}}
    return payload

@app.get("/health")
def health():
    return {
        "modbus_connected": bool(getattr(client, "connected", False)),
        "last_poll_ts": _last_health.get("last_poll_ts"),
        "ok": _last_health.get("ok"),
        "db_queue_size": DB_Q.qsize(),
        "backoff": _last_health.get("backoff"),
        "ws_conns": len(ws_manager.active),
    }

@app.get("/history")
def history(name_full: str = Query(...), since: Optional[float] = Query(None), until: Optional[float] = Query(None), limit: int = Query(1000)):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    sql = "SELECT ts, value FROM readings WHERE name=?"
    params = [name_full]
    if since is not None:
        sql += " AND ts>=?"
        params.append(float(since))
    if until is not None:
        sql += " AND ts<=?"
        params.append(float(until))
    sql += " ORDER BY ts ASC LIMIT ?"
    params.append(int(limit))
    cur.execute(sql, params)
    rows = [{"ts": r[0], "value": r[1]} for r in cur.fetchall()]
    conn.close()
    return rows

def _find_point_by_name(name: str) -> Optional[Dict[str, Any]]:
    for p in points():
        if p["name"] == name:
            return p
    return None

@app.post("/write")
async def write(req: WriteReq):
    # 解析 full_key 或 (name + ids)
    if req.name_full:
        base_name, c_id, m_id, s_id = parse_full_key(req.name_full)
        pt = _find_point_by_name(base_name)
        if pt is None:
            raise HTTPException(404, f"{req.name_full} 不存在（基础名未找到）")
    else:
        if not req.name:
            raise HTTPException(400, "必须提供 name_full 或 (name + cluster_id/module_id/sub_id)")
        pt = _find_point_by_name(req.name)
        if pt is None:
            raise HTTPException(404, f"{req.name} 不存在")
        # 对未用到的层级填 -1，最终 compose 时会以配置中默认ID参与（不建议）
        c_id = req.cluster_id if req.cluster_id is not None else -1
        m_id = req.module_id  if req.module_id  is not None else -1
        s_id = req.sub_id     if req.sub_id     is not None else -1

    # 校验写权限
    if pt.get("rw","R") not in ("W","RW"):
        raise HTTPException(400, f"{pt['name']} 不是可写点")

    slave = int(pt["slave"])
    func  = int(pt["func"])
    typ   = pt.get("type","uint16")
    addr  = compose_addr_with_ids(
        pt,
        c_id if c_id!=-1 else int(CFG.get("cluster_id",0)),
        m_id if m_id!=-1 else int(CFG.get("module_id",0)),
        s_id if s_id!=-1 else int(CFG.get("sub_id",0)),
    )
    regs  = encode_value(typ, req.value, pt.get("scale"))

    if client is None or not getattr(client, "connected", False):
        raise HTTPException(500, "Modbus 未连接")

    if func in (6, 16):
        if func == 6:
            if len(regs) != 1:
                raise HTTPException(400, "功能码 6 仅支持 1 寄存器类型")
            rr = await asyncio.to_thread(client.write_register, addr, regs[0], slave=slave)
        else:
            rr = await asyncio.to_thread(client.write_registers, addr, regs, slave=slave)
        if (not rr) or rr.isError():
            raise HTTPException(502, f"写入失败: {rr}")
        # 更新缓存 & 入库 & 推送
        full_key = build_full_key(pt["name"], c_id, m_id, s_id)
        val = decode_value(typ, regs, pt.get("scale"))
        now = time.time()
        CACHE[full_key] = {"value": val, "ts": now, "raw": regs}
        raw_bytes = b"".join(int(x).to_bytes(2, "big") for x in regs)
        try:
            DB_Q.put_nowait((full_key, now, float(val) if isinstance(val,(int,float)) else None, raw_bytes))
        except asyncio.QueueFull:
            logging.warning("DB 队列满，丢弃写入: %s", full_key)

        payload = {"type":"update","data":{full_key: val}}
        if CFG.get("include_meta", True):
            payload["meta"] = {full_key: {"c": c_id, "m": m_id, "s": s_id}}
        await ws_manager.broadcast(payload)
        return {"ok": True, "name_full": full_key, "value": val}
    else:
        raise HTTPException(400, f"不可用的写功能码: {func}（应为 6 或 16）")

@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        while True:
            _ = await ws.receive_text()  # 前端可发 ping
            await ws.send_text(json.dumps({"type":"pong"}, ensure_ascii=False))
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)

# 心跳
async def _heartbeat_task():
    while True:
        try:
            await ws_manager.broadcast({"type":"heartbeat","ts": time.time()})
        except Exception:
            pass
        await asyncio.sleep(5.0)
