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

read -r -s -p "Gemini API key: " BOTTY_RESEARCH_KEY
echo
if [[ -z ${BOTTY_RESEARCH_KEY} ]]; then
  echo "Value cannot be empty." >&2
  exit 1
fi
export BOTTY_RESEARCH_KEY

python3 - "${ENV_FILE}" <<'PY'
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
key = "RESEARCH_GEMINI_API_KEY"
lines = path.read_text().splitlines()
updated = []
seen = False
for line in lines:
    if line.split("=", 1)[0] == key:
        updated.append(f"{key}={os.environ['BOTTY_RESEARCH_KEY']}")
        seen = True
    else:
        updated.append(line)
if not seen:
    updated.append(f"{key}={os.environ['BOTTY_RESEARCH_KEY']}")
path.write_text("\n".join(updated) + "\n")
PY

chmod 600 "${ENV_FILE}"
unset BOTTY_RESEARCH_KEY
systemctl restart botty
echo "Stock Analyst configured. Botty has been restarted."
