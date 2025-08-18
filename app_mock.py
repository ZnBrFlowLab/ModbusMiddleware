#!/usr/bin/env python
# -*- coding: utf-8 -*-
import asyncio, time, json, yaml, math, random, sqlite3, logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="Modbus 中台（模拟）", version="mock-1.0.0")

CONFIG_PATH = Path(__file__).with_name("config_mock.yaml")
DB_PATH = Path(__file__).with_name("data_mock.sqlite3")

CFG: Dict[str, Any] = {}
TAGMAP: Dict[str, Any] = {}
CACHE: Dict[str, Dict[str, Any]] = {}         # {full_key: {value, ts, raw}}
META: Dict[str, Dict[str, int]] = {}          # {full_key: {c,m,s}}
DB_Q: "asyncio.Queue[Tuple[str, float, Optional[float]]]" = asyncio.Queue(maxsize=20000)

# ============== 工具：范围解析 / 键名 / 元信息 ==============
def _parse_range(spec, fallback: List[int]) -> List[int]:
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
    else:
        for c in clusters:
            for m in modules:
                for s in subs:
                    out.append((c, m, s))
    return out

def build_full_key(name: str, c: int, m: int, s: int) -> str:
    fmt = CFG.get("key_format", "{name}@C{c}M{m}S{s}")
    return fmt.format(name=name, c=c, m=m, s=s)

def parse_full_key(full_key: str) -> Tuple[str,int,int,int]:
    # 仅支持默认格式的反解析
    default_fmt = "{name}@C{c}M{m}S{s}"
    if CFG.get("key_format", default_fmt) != default_fmt:
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

# ============== 配置/点表加载（无点表则自动造一个） ==============
def load_config():
    global CFG
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        CFG = yaml.safe_load(f)

def ensure_tagmap():
    """若找不到 tagmap 文件，则自动生成一个“全字段、常见类型”的示例点表，保证前端有数据。"""
    global TAGMAP
    p = Path(CFG.get("tagmap_path", "generated_tagmap.json")).resolve()
    if p.exists():
        TAGMAP = json.loads(p.read_text(encoding="utf-8"))
        return

    demo = {
        "points": [
            # cluster 级
            {"name":"stack_voltage","slave":1,"func":4,"addr":1,"type":"float32","scale":0.1,"unit":"V","rw":"R","level":"cluster","desc":"电堆电压"},
            {"name":"stack_current","slave":1,"func":4,"addr":3,"type":"float32","scale":0.1,"unit":"A","rw":"R","level":"cluster","desc":"电堆电流"},
            {"name":"stack_temp","slave":1,"func":4,"addr":5,"type":"int16","scale":0.1,"unit":"℃","rw":"R","level":"cluster","desc":"电堆温度"},
            {"name":"soc","slave":1,"func":4,"addr":7,"type":"uint16","scale":0.1,"unit":"%","rw":"R","level":"cluster","desc":"SOC 百分比"},
            # module 级
            {"name":"module_voltage","slave":1,"func":4,"addr":100,"type":"uint16","scale":0.1,"unit":"V","rw":"R","level":"module","desc":"模块电压"},
            {"name":"pump_enable","slave":1,"func":6,"addr":120,"type":"bool","unit":"","rw":"RW","level":"module","desc":"循环泵启停"},
            # substack 级
            {"name":"cell_voltage","slave":1,"func":4,"addr":200,"type":"uint16","scale":0.01,"unit":"V","rw":"R","level":"substack","desc":"子串单体电压"},
            {"name":"cell_temp","slave":1,"func":4,"addr":210,"type":"int16","scale":0.1,"unit":"℃","rw":"R","level":"substack","desc":"子串温度"},
        ]
    }
    p.write_text(json.dumps(demo, ensure_ascii=False, indent=2), encoding="utf-8")
    TAGMAP = demo
    logging.warning("未发现点表，已自动生成示例点表：%s", p)

def points() -> List[Dict[str, Any]]:
    return TAGMAP.get("points", [])

# ============== SQLite（历史持久化） ==============
def init_db():
    conn = sqlite3.connect(DB_PATH)
    with conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            ts   REAL NOT NULL,
            value REAL
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_readings_name_ts ON readings(name, ts)")
    conn.close()

def db_insert_many(rows: List[Tuple[str, float, Optional[float]]]):
    conn = sqlite3.connect(DB_PATH)
    with conn:
        conn.executemany("INSERT INTO readings(name, ts, value) VALUES (?,?,?)", rows)
    conn.close()

async def db_writer_task():
    BATCH = 200
    INTERVAL = 0.5
    buf: List[Tuple[str, float, Optional[float]]] = []
    last = time.time()
    while True:
        try:
            timeout = max(0.0, last + INTERVAL - time.time())
            try:
                item = await asyncio.wait_for(DB_Q.get(), timeout=timeout)
                buf.append(item)
            except asyncio.TimeoutError:
                pass
            if len(buf) >= BATCH or (buf and time.time() - last >= INTERVAL):
                db_insert_many(buf[:])
                buf.clear()
                last = time.time()
        except Exception as e:
            logging.exception("DB 写入失败: %s", e)
            await asyncio.sleep(1.0)

# ============== 数据生成：确保“所有字段都有数据” ==============
def _seed_for_key(k: str) -> int:
    return abs(hash(k)) % (2**31 - 1)

def gen_value_for_point(base_name: str, typ: str, scale: Optional[float], unit: str, t: float, full_key: str) -> Any:
    """
    根据点名/类型/单位生成稳定且有节奏变化的模拟值。
    - 每个 full_key 有独立随机种子，保证不同键数据不同、且重启后大体一致（但含时间项）。
    - 数值考虑 scale 后输出“工程值”。
    """
    random.seed(_seed_for_key(full_key))
    # 基础波形：sin + noise + 梯形
    w = 2 * math.pi / 30.0            # 30s 周期
    base = math.sin(w * t + random.random()*2*math.pi)
    ramp = ((t % 20.0) / 20.0) * 2 - 1  # -1..1
    noise = (random.random() - 0.5) * 0.2

    if "voltage" in base_name:
        val = 500 + 30*base + 5*ramp + noise*2   # 约 500V 左右
        if "cell_voltage" in base_name:
            val = 2.0 + 0.2*base + 0.05*ramp + noise*0.02  # 单体 2V 左右
    elif "current" in base_name:
        val = 80 + 20*base + 5*ramp + noise*3
    elif base_name == "soc":
        # 0~100 循环
        val = ((t % 300) / 300.0) * 100.0
    elif "temp" in base_name:
        val = 25 + 10*base + noise*2
    elif "enable" in base_name or typ == "bool":
        val = ((int(t) // 10) % 2) == 0  # 每 10s 翻转
        return bool(val)
    else:
        val = 100 + 10*base + noise

    # 保留工程值
    return float(val)

def build_all_full_keys_and_meta():
    """基于点表 + 范围，构造全量 full_key，并初始化 META。"""
    META.clear()
    keys = []
    for p in points():
        level = p.get("level","cluster")
        combos = id_combinations_for_level(level)
        for (c, m, s) in combos:
            # 约定：不存在的层级用 -1 占位（只用于前端归属显示）
            cc = c if c != -1 else -1
            mm = m if m != -1 else -1
            ss = s if s != -1 else -1
            full_key = build_full_key(p["name"], cc, mm, ss)
            META[full_key] = {"c": cc, "m": mm, "s": ss}
            keys.append(full_key)
    return keys

def initialize_cache_with_all_fields():
    """初始化所有 full_key 的首帧数据，保证“所有字段都有数据”"""
    keys = build_all_full_keys_and_meta()
    now = time.time()
    for p in points():
        typ = (p.get("type","uint16") or "uint16").lower()
        scale = p.get("scale")
        unit = p.get("unit","")
        level = p.get("level","cluster")
        combos = id_combinations_for_level(level)
        for (c,m,s) in combos:
            cc = c if c != -1 else -1
            mm = m if m != -1 else -1
            ss = s if s != -1 else -1
            full_key = build_full_key(p["name"], cc, mm, ss)
            val = gen_value_for_point(p["name"], typ, scale, unit, now, full_key)
            # 写缓存
            CACHE[full_key] = {"value": val, "ts": now, "raw": []}
    logging.info("初始化完成：共 %d 个点（含范围展开）", len(CACHE))

# ============== WS 管理 ==============
class ConnManager:
    def __init__(self):
        self.active: List[WebSocket] = []
    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        snapshot = {k: v["value"] for k,v in CACHE.items()}
        payload = {"type":"snapshot","data":snapshot}
        if CFG.get("include_meta", True):
            payload["meta"] = META
        await ws.send_text(json.dumps(payload, ensure_ascii=False))
    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)
    async def broadcast(self, payload: Dict[str, Any]):
        text = json.dumps(payload, ensure_ascii=False)
        for ws in list(self.active):
            try:
                await ws.send_text(text)
            except Exception:
                self.disconnect(ws)

ws_manager = ConnManager()

# ============== 模拟轮询：周期更新所有点位 ==============
async def simulate_loop():
    poll_ms = int(CFG.get("poll_interval_ms", 1000))
    eps = float(CFG.get("float_epsilon", 1e-6))
    while True:
        try:
            now = time.time()
            updates = {}
            rows = []
            for full_key, rec in CACHE.items():
                base_name, _, _, _ = parse_full_key(full_key)
                # 找配置点
                p = next((pp for pp in points() if pp["name"] == base_name), None)
                if p is None:
                    continue
                typ = (p.get("type","uint16") or "uint16").lower()
                scale = p.get("scale")
                unit = p.get("unit","")
                # 生成新值
                new_val = gen_value_for_point(base_name, typ, scale, unit, now, full_key)
                old_val = rec["value"]
                changed = False
                if isinstance(new_val, float) and isinstance(old_val, float):
                    changed = abs(new_val - old_val) > eps
                else:
                    changed = new_val != old_val
                if changed:
                    rec["value"] = new_val
                    rec["ts"] = now
                    updates[full_key] = new_val
                    rows.append((full_key, now, float(new_val) if isinstance(new_val,(int,float)) else None))
            if updates:
                payload = {"type":"update","data":updates}
                if CFG.get("include_meta", True):
                    # 仅携带变更项的 meta，或都带上 META 均可
                    payload["meta"] = {k: META.get(k, {}) for k in updates.keys()}
                await ws_manager.broadcast(payload)
                # 入库
                for r in rows:
                    try:
                        DB_Q.put_nowait(r)
                    except asyncio.QueueFull:
                        logging.warning("DB 队列已满，丢弃一批")
        except Exception as e:
            logging.exception("模拟轮询异常: %s", e)
        await asyncio.sleep(poll_ms/1000.0)

# ============== API ==============
class WriteReq(BaseModel):
    # 模拟环境支持两种写法
    name_full: Optional[str] = None
    name: Optional[str] = None
    cluster_id: Optional[int] = None
    module_id: Optional[int] = None
    sub_id: Optional[int] = None
    value: Union[float, int, bool]

@app.on_event("startup")
async def _startup():
    load_config()
    ensure_tagmap()
    init_db()
    initialize_cache_with_all_fields()
    asyncio.create_task(simulate_loop())
    asyncio.create_task(db_writer_task())
    asyncio.create_task(_heartbeat_task())

@app.get("/tags")
def get_tags():
    return points()

@app.get("/values")
def get_values():
    data = {k: v["value"] for k,v in CACHE.items()}
    if CFG.get("include_meta", True):
        return {"data": data, "meta": META}
    return data

@app.get("/values/{name_full}")
def get_value_by_full(name_full: str):
    if name_full not in CACHE:
        raise HTTPException(404, f"{name_full} 不存在")
    payload = CACHE[name_full]
    if CFG.get("include_meta", True):
        return {"data": payload, "meta": {name_full: META.get(name_full, {})}}
    return payload

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

@app.get("/health")
def health():
    return {
        "mock_mode": True,
        "last_update_ts": max((v["ts"] for v in CACHE.values()), default=0.0),
        "points": len(points()),
        "expanded_keys": len(CACHE),
        "ws_conns": len(ws_manager.active),
        "db_queue_size": DB_Q.qsize(),
    }

@app.post("/write")
def write(req: WriteReq):
    # 模拟写：直接覆盖缓存值，并广播 + 入库
    if req.name_full:
        full_key = req.name_full
    else:
        if not req.name:
            raise HTTPException(400, "必须提供 name_full 或 (name + ids)")
        c = req.cluster_id if req.cluster_id is not None else -1
        m = req.module_id  if req.module_id  is not None else -1
        s = req.sub_id     if req.sub_id     is not None else -1
        full_key = build_full_key(req.name, c, m, s)

    if full_key not in CACHE:
        # 如果是合法点名但未展开，则尝试动态创建（以确保“所有字段可写可测”）
        base_name, c, m, s = parse_full_key(full_key)
        p = next((pp for pp in points() if pp["name"] == base_name), None)
        if p is None:
            raise HTTPException(404, f"{full_key} 不存在")
        CACHE[full_key] = {"value": None, "ts": 0.0, "raw": []}
        META[full_key] = {"c": c, "m": m, "s": s}

    now = time.time()
    CACHE[full_key]["value"] = req.value
    CACHE[full_key]["ts"] = now
    # 入库（仅数值）
    try:
        DB_Q.put_nowait((full_key, now, float(req.value) if isinstance(req.value,(int,float)) else None))
    except asyncio.QueueFull:
        logging.warning("DB 队列已满，写入丢弃")
    # 广播
    payload = {"type":"update","data":{full_key: req.value}}
    if CFG.get("include_meta", True):
        payload["meta"] = {full_key: META.get(full_key, {})}
    asyncio.create_task(ws_manager.broadcast(payload))
    return {"ok": True, "name_full": full_key, "value": req.value}

@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        while True:
            _ = await ws.receive_text()  # 前端可发 ping
            await ws.send_text(json.dumps({"type":"pong"}, ensure_ascii=False))
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)

async def _heartbeat_task():
    while True:
        try:
            await ws_manager.broadcast({"type":"heartbeat","ts": time.time()})
        except Exception:
            pass
        await asyncio.sleep(5.0)
