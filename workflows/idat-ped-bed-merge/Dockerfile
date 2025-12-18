FROM ubuntu:latest
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "idat_ped.py"]
