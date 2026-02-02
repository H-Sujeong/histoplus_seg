#!/usr/bin/env bash
set -euo pipefail

log() { echo "[entrypoint] $*"; }

# ---------------------------
# Env from run.sh / devcontainer
# ---------------------------
: "${HOST_USER:=appuser}"   # ignored (we force APP_USER)
: "${HOST_UID:=1000}"
: "${HOST_GID:=1000}"
: "${WORKDIR:=/workspace}"
: "${ENABLE_PASSWORDLESS_SUDO:=1}"

APP_USER="appuser"
APP_UID="${HOST_UID}"
APP_GID="${HOST_GID}"

# ---------------------------
# Guard: must run as root (need usermod/groupmod/chown)
# ---------------------------
if [[ "$(id -u)" != "0" ]]; then
  log "[FATAL] entrypoint must run as root (current uid=$(id -u))."
  exit 1
fi

# ---------------------------
# 0) Ensure gosu exists
# ---------------------------
if ! command -v gosu >/dev/null 2>&1; then
  log "[FATAL] gosu not found. Install gosu in image."
  exit 1
fi

# ---------------------------
# 1) Ensure group for APP_GID exists
#    We prefer group name "appuser" if possible.
# ---------------------------
EXISTING_GROUP_BY_GID="$(getent group "${APP_GID}" | cut -d: -f1 || true)"

if [[ -z "${EXISTING_GROUP_BY_GID}" ]]; then
  log "Creating group ${APP_USER} (gid=${APP_GID})"
  groupadd -g "${APP_GID}" "${APP_USER}"
  EXISTING_GROUP_BY_GID="${APP_USER}"
else
  # If gid belongs to another group name, try rename to appuser when safe
  if [[ "${EXISTING_GROUP_BY_GID}" != "${APP_USER}" ]]; then
    if ! getent group "${APP_USER}" >/dev/null 2>&1; then
      log "Renaming group '${EXISTING_GROUP_BY_GID}' -> '${APP_USER}' for gid=${APP_GID}"
      groupmod -n "${APP_USER}" "${EXISTING_GROUP_BY_GID}" || true
      EXISTING_GROUP_BY_GID="${APP_USER}"
    else
      log "gid=${APP_GID} belongs to group '${EXISTING_GROUP_BY_GID}', and '${APP_USER}' group already exists. Keeping existing group name."
    fi
  fi
fi

# ---------------------------
# 1.5) Ensure APP_USER exists (as a name)
#      (DevDockerfile should create it, but keep it robust.)
# ---------------------------
if ! getent passwd "${APP_USER}" >/dev/null 2>&1; then
  log "User '${APP_USER}' not found. Creating placeholder user first."
  # choose a safe placeholder uid if APP_UID already used
  PLACEHOLDER_UID=9999
  if getent passwd "${PLACEHOLDER_UID}" >/dev/null 2>&1; then
    PLACEHOLDER_UID=9998
  fi
  useradd -m -u "${PLACEHOLDER_UID}" -g "${APP_GID}" -s /bin/bash "${APP_USER}"
fi

# ---------------------------
# 1.6) If Dev Containers already updated UID/GID, skip remap steps completely.
#      (This prevents double-munging when updateUID feature is used.)
# ---------------------------
SKIP_REMAP=0
CUR_UID="$(id -u "${APP_USER}")"
CUR_GID="$(id -g "${APP_USER}")"

if [[ "${CUR_UID}" == "${APP_UID}" && "${CUR_GID}" == "${APP_GID}" ]]; then
  log "UID/GID already match (${APP_UID}:${APP_GID}), skipping remap steps."
  SKIP_REMAP=1
fi

# ---------------------------
# 2) Remap user/group only if needed
# ---------------------------
if [[ "${SKIP_REMAP}" == "0" ]]; then
  # 2-A) Handle UID collision: if APP_UID already used by another user
  EXISTING_USER_BY_UID="$(getent passwd "${APP_UID}" | cut -d: -f1 || true)"

  if [[ -n "${EXISTING_USER_BY_UID}" && "${EXISTING_USER_BY_UID}" != "${APP_USER}" ]]; then
    log "UID ${APP_UID} belongs to '${EXISTING_USER_BY_UID}'. Forcing rename to '${APP_USER}'."

    # If appuser already exists (name conflict), move it aside
    if getent passwd "${APP_USER}" >/dev/null 2>&1; then
      OLD_UID="$(id -u "${APP_USER}")"
      BACKUP_NAME="appuser_old_${OLD_UID}"
      log "User '${APP_USER}' exists (uid=${OLD_UID}). Renaming it to '${BACKUP_NAME}'."
      usermod -l "${BACKUP_NAME}" "${APP_USER}" || true

      # Best-effort home rename
      if [[ -d "/home/${APP_USER}" && ! -d "/home/${BACKUP_NAME}" ]]; then
        mv "/home/${APP_USER}" "/home/${BACKUP_NAME}" || true
      fi
    fi

    # Rename UID owner -> appuser
    usermod -l "${APP_USER}" "${EXISTING_USER_BY_UID}" || true

    # Best-effort: normalize home to /home/appuser
    if [[ -d "/home/${EXISTING_USER_BY_UID}" && ! -d "/home/${APP_USER}" ]]; then
      mv "/home/${EXISTING_USER_BY_UID}" "/home/${APP_USER}" || true
    fi
    usermod -d "/home/${APP_USER}" -m "${APP_USER}" || true
  fi

  # 2-B) Now appuser name must exist
  if ! getent passwd "${APP_USER}" >/dev/null 2>&1; then
    log "[FATAL] Failed to ensure user '${APP_USER}' exists after collision handling."
    exit 1
  fi

  # 2-C) Ensure appuser primary GID matches APP_GID
  CUR_UID="$(id -u "${APP_USER}")"
  CUR_GID="$(id -g "${APP_USER}")"

  if [[ "${CUR_GID}" != "${APP_GID}" ]]; then
    log "Updating primary gid of ${APP_USER}: ${CUR_GID} -> ${APP_GID}"
    usermod -g "${APP_GID}" "${APP_USER}" || true
  fi

  # 2-D) Ensure appuser UID matches APP_UID
  CUR_UID="$(id -u "${APP_USER}")"
  if [[ "${CUR_UID}" != "${APP_UID}" ]]; then
    log "Updating uid of ${APP_USER}: ${CUR_UID} -> ${APP_UID}"
    usermod -u "${APP_UID}" "${APP_USER}"
  fi

  # 2-E) Ensure home dir is /home/appuser
  if [[ "$(getent passwd "${APP_USER}" | cut -d: -f6)" != "/home/${APP_USER}" ]]; then
    log "Setting home for ${APP_USER} to /home/${APP_USER}"
    usermod -d "/home/${APP_USER}" -m "${APP_USER}" || true
  fi
else
  log "Remap skipped."
fi

# ---------------------------
# 3) Sudo policy (optional)
# ---------------------------
if command -v sudo >/dev/null 2>&1; then
  if [[ "${ENABLE_PASSWORDLESS_SUDO}" == "1" ]]; then
    log "Enabling passwordless sudo for ${APP_USER}"
    echo "${APP_USER} ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/99-${APP_USER}"
    chmod 0440 "/etc/sudoers.d/99-${APP_USER}"
  else
    rm -f "/etc/sudoers.d/99-${APP_USER}" || true
  fi
fi

# ---------------------------
# 4) Workspace + HOME permissions (best-effort)
# ---------------------------
mkdir -p "${WORKDIR}"
chown -R "${APP_UID}:${APP_GID}" "${WORKDIR}" || true

mkdir -p "/home/${APP_USER}/.cache" "/home/${APP_USER}/.config" "/home/${APP_USER}/.local/share/jupyter"
chown -R "${APP_UID}:${APP_GID}" "/home/${APP_USER}" || true

# ---------------------------
# 5) Drop privileges
# ---------------------------
cd "${WORKDIR}"
export HOME="/home/${APP_USER}"
export USER="${APP_USER}"

# Make micromamba env visible without activation
export PATH="/opt/micromamba/envs/hist/bin:${PATH}"

# Help Jupyter discover kernels in env + user space
export JUPYTER_PATH="/opt/micromamba/envs/hist/share/jupyter:${JUPYTER_PATH:-}"
export JUPYTER_DATA_DIR="/home/${APP_USER}/.local/share/jupyter"

log "Running as ${APP_USER} (uid=${APP_UID}, gid=${APP_GID})"
exec gosu "${APP_UID}:${APP_GID}" "$@"
