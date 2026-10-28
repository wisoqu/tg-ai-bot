FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --system --create-home --uid 10001 botuser
COPY --chown=botuser:botuser . .
RUN mkdir -p /app/data && chown -R botuser:botuser /app/data
USER botuser
ENV SQLITE_PATH=/app/data/bot_memory.sqlite3
CMD ["python", "main.py"]
