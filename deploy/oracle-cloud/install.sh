#!/usr/bin/env bash
set -Eeuo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "Run this installer with sudo." >&2
  exit 1
fi

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
INSTALL_ROOT=/opt/botty
DATA_ROOT=/var/lib/botty
CONFIG_ROOT=/etc/botty
ENV_FILE=${CONFIG_ROOT}/botty.env

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install --yes docker.io rsync
systemctl enable --now docker

# A small swap file keeps image builds reliable on a 1 GB Always Free VM.
if [[ ! -f /swapfile ]]; then
  fallocate -l 2G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
fi
if ! swapon --show=NAME --noheadings | grep -Fxq /swapfile; then
  swapon /swapfile
fi
if ! grep -Fq '/swapfile none swap sw 0 0' /etc/fstab; then
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

install -d -m 0755 "${INSTALL_ROOT}" "${CONFIG_ROOT}"
install -d -m 0750 -o 10001 -g 10001 "${DATA_ROOT}"
rsync -a --delete \
  --exclude .git \
  --exclude .env \
  --exclude storage \
  "${SOURCE_ROOT}/" "${INSTALL_ROOT}/"

FRESH_ENV=0
if [[ ! -f ${ENV_FILE} ]]; then
  install -m 0600 "${INSTALL_ROOT}/.env.example" "${ENV_FILE}"
  FRESH_ENV=1
fi

docker build --tag botty-oracle:latest "${INSTALL_ROOT}"
install -m 0644 "${INSTALL_ROOT}/deploy/oracle-cloud/botty.service" \
  /etc/systemd/system/botty.service
systemctl daemon-reload

echo
if [[ ${FRESH_ENV} -eq 1 ]]; then
  echo "Botty is installed but has not been started."
  echo "1. Edit ${ENV_FILE} and fill ALPACA_API_KEY, ALPACA_SECRET_KEY,"
  echo "   AUTO_WEBHOOK_URL, and DASHBOARD_PASSWORD."
  echo "2. Run: sudo /opt/botty/deploy/oracle-cloud/start.sh"
else
  # An upgrade keeps the existing environment file, so telling the operator to
  # fill in credentials they set months ago invites them to re-enter secrets
  # that are already correct. Say what actually remains to be done.
  echo "Botty is rebuilt. ${ENV_FILE} was left as it is, so credentials are intact."
  echo "Restart to pick up the new image:"
  echo "  sudo systemctl restart botty"
  echo
  echo "New settings are never written into an existing environment file. To see"
  echo "what became configurable since it was written:"
  echo "  diff <(grep -o '^[A-Z_]*' ${ENV_FILE} | sort -u) \\"
  echo "       <(grep -o '^[A-Z_]*' ${INSTALL_ROOT}/.env.example | sort -u)"
fi
