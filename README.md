# Modbus 模拟服务（app_mock.py）
https://xw5mwxve99.feishu.cn/wiki/BbjhwR0oyiIj4SklIgpcNAcjn1b?from=from_copylink
# 启动服务
uvicorn app:app --host 0.0.0.0 --port 8000

# 手动触发清理（保留 7 天）
curl -X POST "http://127.0.0.1:8000/admin/cleanup?days=7"

# FULL VACUUM（低峰期使用，文件压缩更彻底）
curl -X POST "http://127.0.0.1:8000/admin/compact"

# 导出最近 7 天所有点位到 Excel
python export_readings_to_excel.py --db data.sqlite3 \
  --since "2025-08-12 00:00:00" --out export_last7d.xlsx
