#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STORAGE_DIR="${SCRIPT_DIR}/storage"
PROD_MODE=0
ARGS=()

for arg in "$@"; do
  case "${arg}" in
    --prod)
      PROD_MODE=1
      ;;
    *)
      ARGS+=("${arg}")
      ;;
  esac
done

if [ "${PROD_MODE}" -eq 1 ]; then
  if [ -d "${STORAGE_DIR}" ]; then
    find "${STORAGE_DIR}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  fi
fi

# 直连 PostgreSQL，不再依赖 PgBouncer
if [ "${#ARGS[@]}" -gt 0 ]; then
  jesse run "${ARGS[@]}"
else
  jesse run
fi
