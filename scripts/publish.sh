#!/usr/bin/env bash
set -euo pipefail

DIST_DIR="dist"

echo "[publish] limpando builds antigos..."
rm -rf ${DIST_DIR}/* || true
python -m pip install --upgrade build twine

echo "[publish] build (sdist + wheel)..."
python -m build

echo "[publish] checando metadata (twine check)..."
twine check ${DIST_DIR}/*

if [ "${1:-}" = "test" ]; then
  echo "[publish] upload para TestPyPI..."
  twine upload --repository testpypi ${DIST_DIR}/*
  exit 0
fi

if [ -z "${TWINE_USERNAME:-}" ] || [ -z "${TWINE_PASSWORD:-}" ]; then
  echo "ERRO: defina TWINE_USERNAME e TWINE_PASSWORD (use __token__ e seu pypi-token)."
  exit 2
fi

echo "[publish] upload para PyPI..."
twine upload ${DIST_DIR}/*

echo "[publish] pronto."
