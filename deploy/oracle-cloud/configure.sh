#!/usr/bin/env bash
set -Eeuo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "Run this configuration helper with sudo." >&2
  exit 1
fi

ENV_FILE=/etc/botty/botty.env
if [[ ! -f ${ENV_FILE} ]]; then
  echo "Missing ${ENV_FILE}; run install.sh first." >&2
  exit 1
fi

read_secret() {
  local prompt=$1
  local variable=$2
  local value

  while true; do
    read -r -s -p "${prompt}: " value
    echo
    if [[ -n ${value} ]]; then
      printf -v "${variable}" '%s' "${value}"
      return
    fi
    echo "Value cannot be empty."
  done
}

echo "Enter each value when prompted. Input is hidden and is not saved in shell history."
read_secret "Alpaca paper API key" BOTTY_CONFIG_API_KEY
read_secret "Alpaca paper secret key" BOTTY_CONFIG_SECRET_KEY
read_secret "Slack #general webhook URL" BOTTY_CONFIG_WEBHOOK_URL
read_secret "New dashboard password" BOTTY_CONFIG_DASHBOARD_PASSWORD

export BOTTY_CONFIG_API_KEY BOTTY_CONFIG_SECRET_KEY
export BOTTY_CONFIG_WEBHOOK_URL BOTTY_CONFIG_DASHBOARD_PASSWORD

python3 - "${ENV_FILE}" <<'PY'
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
replacements = {
    "ALPACA_API_KEY": os.environ["BOTTY_CONFIG_API_KEY"],
    "ALPACA_SECRET_KEY": os.environ["BOTTY_CONFIG_SECRET_KEY"],
    "AUTO_WEBHOOK_URL": os.environ["BOTTY_CONFIG_WEBHOOK_URL"],
    "DASHBOARD_PASSWORD": os.environ["BOTTY_CONFIG_DASHBOARD_PASSWORD"],
}

lines = path.read_text().splitlines()
seen = set()
updated = []
for line in lines:
    key = line.split("=", 1)[0]
    if key in replacements:
        updated.append(f"{key}={replacements[key]}")
        seen.add(key)
    else:
        updated.append(line)

missing = replacements.keys() - seen
if missing:
    raise SystemExit(f"Missing expected settings: {', '.join(sorted(missing))}")

path.write_text("\n".join(updated) + "\n")
PY

chmod 600 "${ENV_FILE}"
unset BOTTY_CONFIG_API_KEY BOTTY_CONFIG_SECRET_KEY
unset BOTTY_CONFIG_WEBHOOK_URL BOTTY_CONFIG_DASHBOARD_PASSWORD

echo "Botty credentials saved securely. Trading remains locked to paper mode."
