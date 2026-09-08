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

if [[ ! -f ${ENV_FILE} ]]; then
  install -m 0600 "${INSTALL_ROOT}/.env.example" "${ENV_FILE}"
fi

docker build --tag botty-oracle:latest "${INSTALL_ROOT}"
install -m 0644 "${INSTALL_ROOT}/deploy/oracle-cloud/botty.service" \
  /etc/systemd/system/botty.service
systemctl daemon-reload

echo
echo "Botty is installed but has not been started."
echo "1. Edit ${ENV_FILE} and fill ALPACA_API_KEY, ALPACA_SECRET_KEY,"
echo "   AUTO_WEBHOOK_URL, and DASHBOARD_PASSWORD."
echo "2. Run: sudo /opt/botty/deploy/oracle-cloud/start.sh"
