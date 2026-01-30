#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

IMAGE_NAME="${1:-hist:base}"

docker build -t "${IMAGE_NAME}" .
echo "[OK] built: ${IMAGE_NAME}"
