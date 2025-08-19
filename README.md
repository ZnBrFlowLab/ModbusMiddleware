# Modbus Middleware

Modbus Middleware is a FastAPI-based service that exposes Modbus RTU/TCP devices to web applications. It provides data polling, SQLite storage, REST and WebSocket APIs, and a mock mode for development.

## Features
- Support for Modbus RTU and Modbus TCP
- SQLite-based persistence
- Periodic polling of configured registers
- REST and WebSocket interfaces
- Mock service for environments without hardware
- Export history to Excel

## Installation
1. Clone the repository
   ```bash
   git clone https://github.com/ZnBrFlowLab/modbus_middleware.git
   cd modbus_middleware
   ```
2. Create a virtual environment
   ```bash
   python3 -m venv venv
   source venv/bin/activate    # Linux/macOS
   # venv\\Scripts\\activate      # Windows
   ```
3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Start the service
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
Then browse `http://127.0.0.1:8000/docs` for Swagger UI.

### Start mock service
```bash
uvicorn app_mock:app --host 0.0.0.0 --port 8000
```

### Export readings to Excel
```bash
python export_readings_to_excel.py \
    --since "2024-08-01" \
    --until "2024-08-10" \
    --filter name1 \
    --filter name2
```

## Configuration
Configuration files live at:
- `config.yaml` – production settings
- `config_mock.yaml` – mock settings

Example snippet:
```yaml
mode: "rtu"        # running mode: rtu or tcp

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

polling_interval: 5   # seconds

database:
  file: "data.sqlite3"
```

## Project Structure
```
.
├── app.py                    # Main service entry
├── app_mock.py               # Mock service entry
├── config.yaml               # Real environment configuration
├── config_mock.yaml          # Mock configuration
├── export_readings_to_excel.py  # Data export tool
├── requirements.txt          # Dependency list
└── README.md                 # Project documentation
```

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/xxx`)
3. Commit your changes (`git commit -am 'Add some xxx'`)
4. Push the branch (`git push origin feature/xxx`)
5. Open a Pull Request

## License
MIT

