FROM python:3.9-slim

WORKDIR /app

COPY ./monitoring-service/requirements.txt .

#COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY ../shared /app/shared

COPY ./monitoring-service/* /app/
ENV PYTHONPATH=/app

ENV PORT=8002
ENV DATA_DIR=/data

VOLUME /data

CMD ["python", "app.py"]