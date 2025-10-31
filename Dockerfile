FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye
WORKDIR /work
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt || true
