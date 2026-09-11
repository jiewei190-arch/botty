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

read_secret() {
  local prompt=$1
  local variable=$2
  local expected_prefix=$3
  local value
  while true; do
    read -r -s -p "${prompt}: " value
    echo
    if [[ ${value} == "${expected_prefix}"* ]]; then
      printf -v "${variable}" '%s' "${value}"
      return
    fi
    echo "That token should begin with ${expected_prefix}"
  done
}

echo "Paste the two Slack tokens when prompted. Input is hidden and not saved in history."
read_secret "Slack bot token (xoxb-...)" BOTTY_SLACK_BOT_TOKEN "xoxb-"
read_secret "Slack app token (xapp-...)" BOTTY_SLACK_APP_TOKEN "xapp-"
export BOTTY_SLACK_BOT_TOKEN BOTTY_SLACK_APP_TOKEN

python3 - "${ENV_FILE}" <<'PY'
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
values = {
    "AUTO_SLACK_BOT_TOKEN": os.environ["BOTTY_SLACK_BOT_TOKEN"],
    "AUTO_SLACK_APP_TOKEN": os.environ["BOTTY_SLACK_APP_TOKEN"],
}
lines = path.read_text().splitlines()
found = set()
updated = []
for line in lines:
    key = line.split("=", 1)[0]
    if key in values:
        updated.append(f"{key}={values[key]}")
        found.add(key)
    else:
        updated.append(line)
for key in values.keys() - found:
    updated.append(f"{key}={values[key]}")
path.write_text("\n".join(updated) + "\n")
PY

chmod 600 "${ENV_FILE}"
unset BOTTY_SLACK_BOT_TOKEN BOTTY_SLACK_APP_TOKEN
echo "Slack status tokens saved. Rebuild and restart Botty to enable status replies."
