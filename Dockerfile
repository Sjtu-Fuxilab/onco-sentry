# Dockerfile for ONCO-SENTRY (runtime + tests)
FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1         PYTHONDONTWRITEBYTECODE=1         PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends         git build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY requirements.txt /work/requirements.txt
RUN python -m pip install -U pip wheel &&         if [ -f requirements.txt ]; then pip install -r requirements.txt; fi &&         pip install pytest pytest-cov pip-audit cyclonedx-bom

COPY . /work
CMD ["python", "-V"]
