#!/usr/bin/env python3
"""Repository-root launcher so ``python main.py <command>`` works without installing."""

from trading_bot.main import main

if __name__ == "__main__":
    raise SystemExit(main())
