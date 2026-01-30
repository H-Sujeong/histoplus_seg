#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# 기본값
# ----------------------------
IMAGE_NAME="hist:base"
CONTAINER_NAME=""

# ----------------------------
# 인자 파싱
# ----------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      CONTAINER_NAME="$2"
      shift 2
      ;;
    *)
      IMAGE_NAME="$1"
      shift
      ;;
  esac
done

HOST_USER="$(id -un)"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

# ===== 필수 마운트 목록 =====
MOUNTS=(
  "$PWD"
  "/home"
  "/data"
  "/home/nas2_fast"
)

# ---- 존재 체크 ----
for p in "${MOUNTS[@]}"; do
  if [ ! -e "$p" ]; then
    echo "[ERROR] mount path not found on host: $p" >&2
    exit 1
  fi
done

# ---- docker -v args 생성 ----
V_ARGS=()
for p in "${MOUNTS[@]}"; do
  if [[ "$p" == *" "* ]]; then
    echo "[ERROR] mount path contains spaces: $p" >&2
    exit 1
  fi
  V_ARGS+=("-v" "${p}:${p}")
done

# ----------------------------
# --name 옵션 처리
# ----------------------------
NAME_ARGS=()
if [[ -n "$CONTAINER_NAME" ]]; then
  NAME_ARGS=(--name "$CONTAINER_NAME")
fi

# ----------------------------
# docker run
# ----------------------------
docker run --rm -it \
  --gpus all \
  "${NAME_ARGS[@]}" \
  -e HOST_USER="${HOST_USER}" \
  -e HOST_UID="${HOST_UID}" \
  -e HOST_GID="${HOST_GID}" \
  "${V_ARGS[@]}" \
  -w "$PWD" \
  "${IMAGE_NAME}" \
  bash
