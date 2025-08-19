# Modbus 中间件

Modbus 中间件是一个基于 FastAPI 的服务，用于将 Modbus RTU/TCP 设备暴露给 Web 应用。它提供数据轮询、SQLite 存储、REST 和 WebSocket API 以及用于开发的模拟模式。

## 功能
- 支持 Modbus RTU 和 Modbus TCP
- 基于 SQLite 的持久化
- 定期轮询配置的寄存器
- REST 和 WebSocket 接口
- 在没有硬件的环境下可运行的模拟服务
- 将历史数据导出为 Excel

## 安装
1. 克隆仓库
   ```bash
   git clone https://github.com/ZnBrFlowLab/modbus_middleware.git
   cd modbus_middleware
   ```
2. 创建虚拟环境
   ```bash
   python3 -m venv venv
   source venv/bin/activate    # Linux/macOS
   # venv\\Scripts\\activate    # Windows
   ```
3. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

## 使用

### 启动服务
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
然后访问 `http://127.0.0.1:8000/docs` 查看 Swagger UI。

### 前端接口说明

服务默认监听 `8000` 端口，可通过 `--port` 参数修改。以下接口均在该端口暴露（示例以 `http://localhost:8000` 为例）：

| 方法 | 路径 | 说明 | 返回示例 |
| ---- | ---- | ---- | ---- |
| GET | `/tags` | 获取所有点位及元信息 | `{ "points": [...] }` |
| GET | `/values` | 获取当前所有点位值 | `{ "温度_1": 25.3, ... }` |
| GET | `/values/{name_full}` | 获取指定点位值 | `{ "value": 25.3, "ts": 1710000000 }` |
| GET | `/history` | 查询历史数据，需传 `name_full`、`since`、`until` 查询参数 | `{ "data": [{"ts":..., "value":...}, ...] }` |
| POST | `/write` | 写入点位值，Body: `{ "name_full": "...", "value": ... }` | `{ "ok": true }` |
| WebSocket | `/ws` | 建立 WebSocket 连接获取实时更新 | `{ "type": "update", "data": {"温度_1": 25.3} }` |

前端可通过 `http://<服务器地址>:8000/路径` 发送 HTTP 请求，或使用 `ws://<服务器地址>:8000/ws` 建立 WebSocket 连接。

### 启动模拟服务
```bash
uvicorn app_mock:app --host 0.0.0.0 --port 8000
```

### 将读数导出为 Excel
```bash
python export_readings_to_excel.py \
    --since "2024-08-01" \
    --until "2024-08-10" \
    --filter name1 \
    --filter name2
```

## 配置
配置文件位于：
- `config.yaml` – 生产环境设置
- `config_mock.yaml` – 模拟环境设置

示例片段：
```yaml
mode: "rtu"        # 运行模式：rtu 或 tcp

rtu:
  port: "/dev/ttyUSB0"
  baudrate: 9600
  parity: "N"
  stopbits: 1
  bytesize: 8
  timeout: 1.0

tcp:
  host: "127.0.0.1"
  port: 502
  timeout: 1.0

polling_interval: 5   # 秒

database:
  file: "data.sqlite3"
```

## 项目结构
```
.
├── app.py                    # 服务入口
├── app_mock.py               # 模拟服务入口
├── config.yaml               # 实际环境配置
├── config_mock.yaml          # 模拟环境配置
├── export_readings_to_excel.py  # 数据导出工具
├── requirements.txt          # 依赖列表
└── README.md                 # 项目文档
```

## 贡献
1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/xxx`)
3. 提交更改 (`git commit -am 'Add some xxx'`)
4. 推送分支 (`git push origin feature/xxx`)
5. 打开 Pull Request

## 许可证
MIT

