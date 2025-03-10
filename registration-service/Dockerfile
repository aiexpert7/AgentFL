FROM python:3.9-slim

WORKDIR /app

COPY ./registration-service/requirements.txt .


#COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./registration-service/* /app/

COPY . .
COPY ../shared /app/shared

ENV PYTHONPATH=/app

ENV PORT=8001
ENV DATA_DIR=/data

VOLUME /data

CMD ["python", "app.py"]