FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml requirements.txt README.md ./
COPY trading_bot ./trading_bot
COPY main.py ./main.py
RUN pip install --no-cache-dir .

RUN useradd --create-home --uid 10001 botty \
    && mkdir -p /app/storage /app/logs \
    && chown -R botty:botty /app
USER botty

HEALTHCHECK --interval=5m --timeout=20s --start-period=2m --retries=3 \
    CMD ["python", "main.py", "status", "--max-heartbeat-age", "15"]

CMD ["python", "main.py", "hunt", "--watch-market", "--paper-trade"]
