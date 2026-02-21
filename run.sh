#!/bin/bash
BASE_DIR=$(dirname "$(realpath "$0")")
set -a
source .env
set +a
echo $JAVA_HOME
source "$BASE_DIR/airflow_venv/bin/activate"
/usr/bin/env python -m airflow standalone
