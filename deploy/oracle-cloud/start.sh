#!/usr/bin/env bash
set -Eeuo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "Run this command with sudo." >&2
  exit 1
fi

ENV_FILE=/etc/botty/botty.env
if [[ ! -f ${ENV_FILE} ]]; then
  echo "Missing ${ENV_FILE}; run install.sh first." >&2
  exit 1
fi

required=(ALPACA_API_KEY ALPACA_SECRET_KEY AUTO_WEBHOOK_URL DASHBOARD_PASSWORD)
for key in "${required[@]}"; do
  value=$(sed -n "s/^${key}=//p" "${ENV_FILE}" | tail -n 1)
  if [[ -z ${value} ]]; then
    echo "Set ${key} in ${ENV_FILE} before starting Botty." >&2
    exit 1
  fi
done

systemctl enable --now botty
sleep 3
systemctl --no-pager --full status botty
