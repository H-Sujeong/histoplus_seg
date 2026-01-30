#!/usr/bin/env bash
set -euo pipefail

# Run-time user mapping:
# - create a user matching HOST_UID/HOST_GID
# - drop privileges with gosu
#
# Required envs (recommended to pass from docker run):
#   HOST_USER, HOST_UID, HOST_GID
#
# Fallbacks are safe defaults.

HOST_USER="${HOST_USER:-appuser}"
HOST_UID="${HOST_UID:-1000}"
HOST_GID="${HOST_GID:-1000}"

# Ensure group exists (by GID). Use a stable name if group name conflicts.
if ! getent group "${HOST_GID}" >/dev/null 2>&1; then
  groupadd -g "${HOST_GID}" "${HOST_USER}" 2>/dev/null || groupadd -g "${HOST_GID}" "grp_${HOST_GID}"
fi

# Ensure user exists (by name). If name exists with different UID, create alt name.
if id -u "${HOST_USER}" >/dev/null 2>&1; then
  true
else
  useradd -m -u "${HOST_UID}" -g "${HOST_GID}" -s /bin/bash "${HOST_USER}" 2>/dev/null \
    || useradd -m -u "${HOST_UID}" -g "${HOST_GID}" -s /bin/bash "u${HOST_UID}"
  if ! id -u "${HOST_USER}" >/dev/null 2>&1; then
    HOST_USER="u${HOST_UID}"
  fi
fi

# Ensure workspace exists; chown best-effort (mounted volumes may not allow)
mkdir -p /workspace
chown -R "${HOST_UID}:${HOST_GID}" /workspace 2>/dev/null || true

# Base env is preinstalled in image. We do NOT install packages here.
export PATH="/opt/micromamba/envs/hist/bin:${PATH}"
export LD_LIBRARY_PATH="/opt/micromamba/envs/hist/lib:${LD_LIBRARY_PATH:-}"

# Nice-to-have: show who we are (comment out if noisy)
echo "[entrypoint] user=${HOST_USER} uid=${HOST_UID} gid=${HOST_GID}"
echo "[entrypoint] workdir=/workspace"

# Drop privileges
exec gosu "${HOST_UID}:${HOST_GID}" "$@"
