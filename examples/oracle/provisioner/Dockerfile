FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir oci>=2.170 oracledb>=3.4.2 kubernetes>=30.0

COPY provision_adb.py /app/provision_adb.py
COPY bootstrap.sql /app/bootstrap.sql

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "/app/provision_adb.py"]
