FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY docker_stats_exporter.py .

EXPOSE 9417

CMD ["python", "docker_stats_exporter.py"]
