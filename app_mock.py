#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FastAPI 模拟中台 · JSON 点表驱动版
─────────────────────────────────
特性：
1) 仅从点表 JSON 生成并约束数据（不再内置英文示例点名）。
2) 按“desc/scale/type/level”等自动推断生成规则：
   - 带“状态/报警/码/阀/液流/负载”等字段 → 生成离散枚举/位掩码值。
   - 标注“保留小数点后1位/scale=0.1” → 输出 1 位小数的工程值。
   - “整数/数量/节数/时间” → 生成稳定整数并合理递增（时间类单调递增）。
   - 组串/模块/子堆 SOC/电压/电流/功率/温度 → 依据当前“组串状态”联动变化。
3) 组串状态严格遵从：0 待机；1 充电；2 放电；3 维护（深度放电）。
4) WS 首发快照 + 增量广播；SQLite 持久化（仅数值型）。

使用：
  # 安装依赖
  pip install fastapi uvicorn pyyaml

  # 准备配置文件 config_mock.yaml（示例）：
  # ----------------------------------------
  # poll_interval_ms: 1000
  # include_meta: true
  # key_format: "{name}@C{c}M{m}S{s}"
  # tagmap_path: "points.json"   # ← 指向你的点表 JSON 文件
  # ranges:
  #   clusters: 0
  #   modules:  0-3
  #   subs:     0-1
  # ----------------------------------------

  uvicorn app_mock_json_driven:app --host 0.0.0.0 --port 8000
  浏览器打开 http://127.0.0.1:8000/docs

说明：
- 点表 JSON 需包含字段：name/slave/func/addr/type/rw/scale/desc/level（与用户提供一致）。
- 若 desc 中出现“0：xx；1：yy；...”的枚举描述，将被解析为合法取值集合；
  对于形如 0x0001/0X0100 的位掩码描述，将按位随机/周期性置位（模拟告警事件）。
- 冷水机运行状态：遵循“0：停机；X：设定的温度；”语义：
  在温度偏高时输出一个设定温度（18~25），否则为 0。
"""
import asyncio, time, json, yaml, math, random, sqlite3, logging, re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from pydantic import BaseModel

# ===================== 基础 ===================== #
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
app = FastAPI(title="Modbus 中台（JSON 驱动模拟）", version="mock-2.0.0")

CONFIG_PATH = Path(__file__).with_name("config_mock.yaml")
DB_PATH     = Path(__file__).with_name("data_mock.sqlite3")

CFG: Dict[str, Any]   = {}
TAGMAP: Dict[str, Any] = {}
CACHE: Dict[str, Dict[str, Any]] = {}   # {full_key: {value, ts}}
META:  Dict[str, Dict[str, int]] = {}   # {full_key: {c,m,s}}
DB_Q:  "asyncio.Queue[Tuple[str, float, Optional[float]]]]" = asyncio.Queue(maxsize=20000)

# 运行期全局模拟状态（驱动联动字段）
SIM: Dict[str, Any] = {
    "start_ts": time.time(),
    "state": 0,                # 0待机 1充电 2放电 3维护
    "state_changed": time.time(),
    "phase_len_s": 45.0,       # 每个状态维持秒数（可调）
    "soc": 60.0,               # %
    "soh": 98.5,               # %
    "voltage": 520.0,          # V（组串级）
    "current": 0.0,            # A（正值）
    "temp": 28.0,              # ℃（组串平均温度）
    "module_count": 8,         # 模块数量
    "cells": 240,              # 节数
}

# ===================== 小工具 ===================== #
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

# 生成稳定随机（不同 full_key 稳定不同）
def _seed_for_key(k: str) -> int:
    return abs(hash(k)) % (2**31 - 1)

# 解析 desc 中的“枚举/位掩码/提示”
#  - 返回 (enums, bitmasks) ：enums 是允许的整数集合；bitmasks 是可能置位的掩码集合
ENUM_PAT = re.compile(r"(?P<key>(?:0x|0X)?[0-9A-Fa-f]+)\s*：")
INT_PAT  = re.compile(r"(?<![0-9A-Fa-f])(\d+)(?![0-9A-Fa-f])")

def parse_desc(desc: Optional[str]) -> Tuple[List[int], List[int]]:
    if not desc:
        return [], []
    enums: List[int] = []
    masks: List[int] = []
    # 捕获形如 0x0001/0X0100
    for m in re.finditer(r"0[xX][0-9A-Fa-f]+", desc):
        masks.append(int(m.group(0), 16))
    # 捕获形如 0：xxx；1：yyy；2：zzz；
    for m in re.finditer(r"(?<![0-9A-Fa-f])(\d+)\s*：", desc):
        enums.append(int(m.group(1)))
    # 如果只有“X：设定温度”这种，留空交给特例处理
    return sorted(set(enums)), sorted(set(masks))

# 层级展开

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
    else:  # substack
        for c in clusters:
            for m in modules:
                for s in subs:
                    out.append((c, m, s))
    return out

# full_key 规则

def build_full_key(name: str, c: int, m: int, s: int) -> str:
    fmt = CFG.get("key_format", "{name}@C{c}M{m}S{s}")
    return fmt.format(name=name, c=c, m=m, s=s)

# ===================== 配置 / 点表 ===================== #

def load_config():
    global CFG
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            CFG = yaml.safe_load(f) or {}
    else:
        # 无配置时提供合理默认
        CFG = {
            "poll_interval_ms": 1000,
            "include_meta": True,
            "key_format": "{name}@C{c}M{m}S{s}",
            "tagmap_path": "points.json",  # 请将点表 JSON 放在同目录并命名为 points.json
            "ranges": {"clusters": 0, "modules": "0-3", "subs": "0-1"},
        }
        logging.warning("未找到 config_mock.yaml，已使用默认配置")


def load_tagmap():
    global TAGMAP
    p = Path(CFG.get("tagmap_path", "points.json")).resolve()
    if not p.exists():
        raise FileNotFoundError(f"点表 JSON 文件不存在：{p}")
    TAGMAP = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(TAGMAP, dict) or "points" not in TAGMAP:
        raise ValueError("点表 JSON 结构非法：根需包含 'points' 数组")


def points() -> List[Dict[str, Any]]:
    return TAGMAP.get("points", [])

# ===================== 数据库 ===================== #

def init_db():
    conn = sqlite3.connect(DB_PATH)
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                ts   REAL NOT NULL,
                value REAL
            )
            """
        )
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

# ===================== 生成规则（核心） ===================== #

# 依据“组串状态”演进全局模拟量

def _step_global(now: float, dt: float):
    # 状态机：每 phase_len_s 轮换 0→1→2→3→0 ...
    if now - SIM["state_changed"] >= SIM["phase_len_s"]:
        SIM["state"] = (SIM["state"] + 1) % 4
        SIM["state_changed"] = now

    st = SIM["state"]

    # SOC 演进：充电上升、放电下降、待机/维护缓变
    ds = 0.0
    if st == 1:   # 充电
        ds = +0.04 * dt   # %/s
    elif st == 2: # 放电
        ds = -0.05 * dt
    elif st == 3: # 维护（深放）
        ds = -0.02 * dt
    else:         # 待机
        ds = +0.005 * dt
    SIM["soc"] = max(0.0, min(100.0, SIM["soc"] + ds))

    # 电流：充电/放电更大，待机很小；维护中等偏低
    target_I = {0: 5.0, 1: 120.0, 2: 110.0, 3: 40.0}[st]
    SIM["current"] += (target_I - SIM["current"]) * min(1.0, dt * 0.5)

    # 电压：随 SOC 略升；维护略低；
    base_V = 450.0 + SIM["soc"] * 0.9  # 450~540
    if st == 3:
        base_V -= 15.0
    SIM["voltage"] += (base_V - SIM["voltage"]) * min(1.0, dt * 0.4)

    # 温度：充/放时升高，待机回落
    target_T = 24.0 + {0: 2.0, 1: 7.0, 2: 6.0, 3: 4.0}[st]
    SIM["temp"] += (target_T - SIM["temp"]) * min(1.0, dt * 0.2)

    # SOH 轻微缓变
    SIM["soh"] = max(85.0, min(100.0, SIM["soh"] + random.uniform(-0.002, 0.002)))

# 生成字段值（工程值），严格遵循点名/desc/scale/type

def gen_value(name: str, p: Dict[str, Any], full_key: str, now: float) -> Any:
    typ   = (p.get("type", "uint16") or "uint16").lower()
    scale = p.get("scale")
    desc  = p.get("desc") or ""
    level = p.get("level", "cluster")

    enums, masks = parse_desc(desc)

    # ======== 特例优先（语义字段） ======== #
    if name in ("组串状态", "模块状态"):
        # 0：待机；1：充电；2：放电；3：维护（深度放电）
        val = SIM["state"]
        return int(val)

    if name == "冷水机运行状态":
        # 0 停机；X 设定温度（在温度较高时给出一个 18~25 的目标值）
        return int(0 if SIM["temp"] < 28.0 else random.randint(18, 25))

    if name == "阀门状态":
        # 0 关闭；1 开启
        return int(((int(now) // 10) % 2) == 1)

    if name == "液流状态":
        # 0X0100 左；0X0001 右；0X0101 两侧
        choices = [0x0100, 0x0001, 0x0101]
        random.seed(_seed_for_key(full_key) + int(now) // 8)
        return int(random.choice(choices))

    if name == "漏液状态":
        # 少量概率触发
        return int(1 if random.random() < 0.02 else 0)

    if name == "放电负载状态":
        # 放电段置 1，其余 0
        return int(1 if SIM["state"] == 2 else 0)

    # 报警码（按位掩码随机闪烁）
    if name in ("1级报警码", "2级报警码", "3级报警码", "组串报警信息"):
        if masks:
            # 低概率置若干位
            on = 0
            for m in masks:
                if random.random() < 0.05:
                    on |= m
            return int(on)
        else:
            # 无掩码时退化为 0/1 报警
            return int(1 if random.random() < 0.05 else 0)

    # 数量/节数/容量等保持稳定整数
    if any(k in name for k in ("数量", "节数")) or ("累计运行时间" not in name and name.endswith("容量")):
        random.seed(_seed_for_key(full_key))
        base = random.randint(4, 16)
        if "模块" in name and "数量" in name:
            return int(SIM["module_count"])  # 与 SIM 同步
        if "电堆节数" in name:
            return int(SIM["cells"])        # 与 SIM 同步
        return int(base)

    # 累计/本次运行时间（秒/分/时/天）
    if name.startswith("累计运行时间") or name.startswith("本次运行时间"):
        # 累计：从服务启动算；本次：从当前状态开始算
        base_ts = SIM["start_ts"] if name.startswith("累计") else SIM["state_changed"]
        seconds = max(0, int(now - base_ts))
        if "（秒）" in name:
            return int(seconds)
        if "（分）" in name:
            return int(seconds // 60)
        if "（时）" in name:
            return int(seconds // 3600)
        if "（天）" in name:
            return int(seconds // 86400)
        return int(seconds)

    # Ah 相关
    if name in ("累计充电Ah", "累计放电Ah", "本次充电Ah", "本次放电Ah"):
        # 简化：按电流比例与状态积分（只模拟，不严格物理）
        dt = CFG.get("poll_interval_ms", 1000) / 1000.0
        factor = SIM["current"] * dt / 3600.0  # A·s → Ah
        key_base = "charge" if "充电" in name else "discharge"
        # 以 full_key 为种子，在 CACHE 的 raw 缓存中滚动累计
        rawkey = f"__{key_base}_ah__"
        store = CACHE.setdefault(full_key, {"value": 0.0, "ts": now})
        acc = store.get(rawkey, 0.0)
        if key_base == "charge" and SIM["state"] == 1:
            acc += factor
        if key_base == "discharge" and SIM["state"] == 2:
            acc += factor
        # “本次”在状态切换时清零
        if name.startswith("本次"):
            if store.get("__last_state__") != SIM["state"]:
                acc = 0.0
        store[rawkey] = acc
        store["__last_state__"] = SIM["state"]
        val = acc * 100.0 / 100.0  # 直接返回 acc
        return round(val, 1)

    # SOC / SOH
    if name in ("组串SOC", "电堆SOC", "子堆SOC"):
        return round(SIM["soc"], 1)
    if name in ("组串SOH",):
        return round(SIM["soh"], 1)

    # 电压 / 电流 / 功率 / 温度 等连续量
    if any(k in name for k in ("电压",)):
        # 组串电压/模块平均电压 等：在 SIM["voltage"] 基础上加少量扰动
        jitter = (math.sin(now/7.0) + random.uniform(-0.5, 0.5)) * 1.5
        val = max(0.0, SIM["voltage"] + jitter)
        return round(val, 1) if scale == 0.1 else float(val)

    if any(k in name for k in ("电流",)):
        jitter = (math.sin(now/5.0) + random.uniform(-0.5, 0.5)) * 3.0
        val = max(0.0, SIM["current"] + jitter)
        return round(val, 1) if scale == 0.1 else float(val)

    if any(k in name for k in ("功率",)):
        p = (SIM["voltage"] * SIM["current"]) / 1000.0  # kW
        p += random.uniform(-0.8, 0.8)
        p = max(0.0, p)
        return round(p, 1) if scale == 0.1 else float(p)

    if any(k in name for k in ("温度",)):
        jitter = (math.sin(now/11.0) + random.uniform(-0.4, 0.4)) * 0.8
        val = SIM["temp"] + jitter
        return round(val, 1) if scale == 0.1 else int(val)

    # 纯枚举（desc 有 0/1/...）且非上面特例时：从枚举集合中取值
    if enums:
        random.seed(_seed_for_key(full_key) + int(now) // 12)
        return int(random.choice(enums))

    # 位掩码字段：偶发置位
    if masks:
        on = 0
        for m in masks:
            if random.random() < 0.05:
                on |= m
        return int(on)

    # 其他：给出一个“看起来合理”的正数
    random.seed(_seed_for_key(full_key) + int(now))
    val = 100.0 + random.random() * 50.0
    return round(val, 1) if scale == 0.1 else float(val)

# 展开全量 full_key 并初始化缓存

def build_all_full_keys_and_meta():
    META.clear()
    keys = []
    for p in points():
        level = p.get("level", "cluster")
        for (c,m,s) in id_combinations_for_level(level):
            full_key = build_full_key(p["name"], c if c!=-1 else -1, m if m!=-1 else -1, s if s!=-1 else -1)
            META[full_key] = {"c": c, "m": m, "s": s}
            keys.append((full_key, p))
    return keys


def initialize_cache_with_all_fields():
    keys = build_all_full_keys_and_meta()
    now = time.time()
    for full_key, p in keys:
        val = gen_value(p["name"], p, full_key, now)
        CACHE[full_key] = {"value": val, "ts": now}
    logging.info("初始化完成：共 %d 个点（含范围展开）", len(CACHE))

# ===================== WS 管理 ===================== #
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

# ===================== 主循环 ===================== #
async def simulate_loop():
    poll_ms = int(CFG.get("poll_interval_ms", 1000))
    eps = float(CFG.get("float_epsilon", 1e-6))
    last = time.time()
    while True:
        try:
            now = time.time()
            dt  = max(0.0, now - last)
            last = now
            _step_global(now, dt)

            updates = {}
            rows: List[Tuple[str, float, Optional[float]]] = []
            for full_key, rec in list(CACHE.items()):
                # 找点定义
                base_name = full_key.split("@",1)[0]
                p = next((pp for pp in points() if pp["name"] == base_name), None)
                if not p:
                    continue
                # 新值
                new_val = gen_value(base_name, p, full_key, now)
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
        await asyncio.sleep(poll_ms / 1000.0)

# ===================== API ===================== #
class WriteReq(BaseModel):
    name_full: Optional[str] = None
    name: Optional[str] = None
    cluster_id: Optional[int] = None
    module_id: Optional[int] = None
    sub_id: Optional[int] = None
    value: Union[float, int, bool]

@app.on_event("startup")
async def _startup():
    load_config()
    load_tagmap()
    # 初始化 SIM 中的常量来自点表（如果有的话）
    try:
        mc = next((p for p in points() if p["name"] == "组串中模块的数量"), None)
        if mc:
            SIM["module_count"] = int(8)
        ec = next((p for p in points() if p["name"] == "电堆节数"), None)
        if ec:
            SIM["cells"] = int(240)
    except Exception:
        pass

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
        sql += " AND ts>=?"; params.append(float(since))
    if until is not None:
        sql += " AND ts<=?"; params.append(float(until))
    sql += " ORDER BY ts ASC LIMIT ?"; params.append(int(limit))
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
        "state": SIM["state"],
        "soc": SIM["soc"],
    }

@app.post("/write")
def write(req: WriteReq):
    # 直接覆盖缓存值，并广播 + 入库
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
        # 动态创建（确保所有字段可被写入）
        base = full_key.split("@",1)[0]
        p = next((pp for pp in points() if pp["name"] == base), None)
        if p is None:
            raise HTTPException(404, f"{full_key} 不存在")
        META[full_key] = {"c": req.cluster_id or -1, "m": req.module_id or -1, "s": req.sub_id or -1}
        CACHE[full_key] = {"value": None, "ts": 0.0}

    now = time.time()
    CACHE[full_key]["value"] = req.value
    CACHE[full_key]["ts"] = now
    try:
        DB_Q.put_nowait((full_key, now, float(req.value) if isinstance(req.value,(int,float)) else None))
    except asyncio.QueueFull:
        logging.warning("DB 队列已满，写入丢弃")

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
