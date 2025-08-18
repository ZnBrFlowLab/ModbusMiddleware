#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
uvicorn app_mock:app --host 0.0.0.0 --port 8000 --reload
