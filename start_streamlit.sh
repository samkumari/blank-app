#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

echo "Root: $ROOT_DIR"

if [ ! -x "$PYTHON" ]; then
  echo "Creating virtual environment at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

echo "Installing requirements into venv..."
"$PIP" install --upgrade pip
"$PIP" install -r "$ROOT_DIR/requirements.txt"

echo "Launching Streamlit with venv python..."
exec "$PYTHON" -m streamlit run "$ROOT_DIR/streamlit_app.py" --server.port=8501
