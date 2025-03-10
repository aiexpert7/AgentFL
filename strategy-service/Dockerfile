FROM python:3.9-slim

WORKDIR /app

COPY ./strategy-service/requirements.txt .

#COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./strategy-service/* /app/

COPY . .
COPY ../shared /app/shared

ENV PYTHONPATH=/app

ENV PORT=8003
ENV DATA_DIR=/data

VOLUME /data

CMD ["python", "app.py"]