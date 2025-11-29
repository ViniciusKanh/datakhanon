#!/usr/bin/env bash
set -euo pipefail

# USO (local): export TWINE_USERNAME="__token__"; export TWINE_PASSWORD="pypi-..."; ./scripts/publish.sh
# Ou use em CI fornecendo TWINE_PASSWORD como secret.

DIST_DIR="dist"

echo "[publish] limpando builds antigos..."
rm -rf ${DIST_DIR}/*
python -m pip install --upgrade build twine

echo "[publish] build (sdist + wheel)..."
python -m build

echo "[publish] checando metadata (twine check)..."
twine check ${DIST_DIR}/*

if [ "${1:-}" = "test" ]; then
  echo "[publish] upload para TestPyPI..."
  twine upload --repository testpypi ${DIST_DIR}/*
  echo "[publish] Para instalar do TestPyPI:"
  echo "  python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple datakhanon"
  exit 0
fi

# Publicar no PyPI real
if [ -z "${TWINE_USERNAME:-}" ] || [ -z "${TWINE_PASSWORD:-}" ]; then
  echo "ERRO: defina TWINE_USERNAME e TWINE_PASSWORD (use __token__ e seu pypi-token)."
  exit 2
fi

echo "[publish] upload para PyPI..."
twine upload ${DIST_DIR}/*

echo "[publish] pronto."
