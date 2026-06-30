#!/bin/bash
BASE_DIR=$(dirname "$(realpath "$0")")
set -a
source .env
set +a
source "$BASE_DIR/airflow_venv/bin/activate"
/usr/bin/env python -m airflow standalone
