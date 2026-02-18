#!/bin/bash
BASE_DIR=$(dirname "$(realpath "$0")")
export AIRFLOW_HOME="$BASE_DIR/airflow"
source "$BASE_DIR/.venv/bin/activate"
/usr/bin/env python -m airflow standalone
