FROM python:3.9-slim

WORKDIR /app

COPY ./orchestrator-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./orchestrator-service/* /app/

COPY . .
COPY ../shared /app/shared

ENV PYTHONPATH=/app

ENV PORT=8004
ENV DATA_DIR=/data
ENV REGISTRATION_SERVICE_URL=http://registration-service:8001
ENV MONITORING_SERVICE_URL=http://monitoring-service:8002
ENV STRATEGY_SERVICE_URL=http://strategy-service:8003

VOLUME /data

CMD ["python", "app.py"]