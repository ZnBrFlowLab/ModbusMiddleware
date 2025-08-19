# Modbus Middleware

一个基于 **FastAPI** 的中台服务，用于通过 **Modbus 协议** 读取电堆参数，并为前端 Web 提供接口。  
支持 **RTU / TCP 模式**，支持 **SQLite 数据存储**，同时提供模拟服务（Mock）以便前端调试。

---

## ✨ 功能特性
- 📡 **协议支持**：支持 Modbus RTU 与 Modbus TCP
- 📊 **数据存储**：SQLite 本地数据库
- 🔄 **轮询机制**：自动轮询寄存器，缓存点位信息
- 🖥 **接口支持**：REST API 与 WebSocket 双通道
- 🧪 **模拟模式**：无硬件环境下可通过 Mock 服务模拟数据
- 📂 **导出工具**：支持将历史数据按点位导出为 Excel

---

## 📦 安装

### 1. 克隆仓库
bash

git clone https://github.com/ZnBrFlowLab/modbus_middleware.git

cd modbus_middleware

### 2. 安装依赖
建议使用虚拟环境：
bash

创建虚拟环境
python3 -m venv venv

激活环境
source venv/bin/activate # Linux/macOS

venv\Scripts\activate # Windows

安装依赖
pip install -r requirements.txt

---

## 🚀 使用方法

### 启动主服务
bash

uvicorn app:app --host 0.0.0.0 --port 8000

启动后访问 http://127.0.0.1:8000/docs 进入 Swagger UI 进行接口调试。

### 启动模拟服务
bash

uvicorn app_mock:app --host 0.0.0.0 --port 8000

### 导出数据为 Excel
bash

python export_readings_to_excel.py \

--since "2024-08-01" \

--until "2024-08-10" \

--filter name1 \

--filter name2

---

## ⚙️ 配置说明
配置文件位于：
- `config.yaml`（真实环境）
- `config_mock.yaml`（模拟环境）

配置示例：
yaml

运行模式 (rtu/tcp)
mode: "rtu"

RTU 配置
rtu:

port: "/dev/ttyUSB0"

baudrate: 9600

parity: "N"

stopbits: 1

bytesize: 8

timeout: 1.0

TCP 配置
tcp:

host: "127.0.0.1"

port: 502

timeout: 1.0

轮询间隔（秒）
polling_interval: 5

数据库配置
database:

file: "data.sqlite3"

---

## 📂 项目结构
bash

.

├── app.py # 主服务入口

├── app_mock.py # 模拟服务入口

├── config.yaml # 真实环境配置

├── config_mock.yaml # 模拟环境配置

├── export_readings_to_excel.py # 数据导出工具

├── requirements.txt # 依赖清单

└── README.md # 项目说明

---

## 🤝 贡献
欢迎通过 Issues 提交问题或通过 Pull Request 贡献代码：
1. Fork 项目仓库
2. 创建功能分支 (`git checkout -b feature/xxx`)
3. 提交修改 (`git commit -am 'Add some xxx'`)
4. 推送分支 (`git push origin feature/xxx`)
5. 创建 Pull Request

---

*License: MIT | Copyright © 2024 ZnBrFlowLab*