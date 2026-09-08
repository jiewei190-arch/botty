"""Run the Botty worker and Streamlit dashboard in one hosted service."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time


def main() -> int:
    """Keep both processes alive; exit if either fails so the host restarts both."""
    port = os.getenv("PORT", "8501")
    processes = [
        subprocess.Popen(  # noqa: S603 - fixed, repository-owned command
            [sys.executable, "main.py", "hunt", "--watch-market", "--paper-trade"]
        ),
        subprocess.Popen(  # noqa: S603 - fixed, repository-owned command
            [
                sys.executable, "-m", "streamlit", "run",
                "trading_bot/dashboard/app.py", "--server.address=0.0.0.0",
                f"--server.port={port}", "--server.headless=true",
            ]
        ),
    ]

    def stop(_signum=None, _frame=None) -> None:
        for process in processes:
            if process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while True:
            for process in processes:
                code = process.poll()
                if code is not None:
                    stop()
                    for child in processes:
                        with contextlib.suppress(subprocess.TimeoutExpired):
                            child.wait(timeout=15)
                    return code or 1
            time.sleep(1)
    finally:
        stop()


if __name__ == "__main__":
    raise SystemExit(main())
