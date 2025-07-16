#!/bin/bash

# Start monitoring services (Prometheus and Grafana)

set -e

echo "Starting monitoring services..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Docker is not running. Please start Docker first."
    exit 1
fi

# Start Prometheus
echo "Starting Prometheus..."
docker run -d \
    --name trader-prometheus \
    -p 9090:9090 \
    -v $(pwd)/docker/prometheus.yml:/etc/prometheus/prometheus.yml \
    prom/prometheus

# Start Grafana
echo "Starting Grafana..."
docker run -d \
    --name trader-grafana \
    -p 3000:3000 \
    -e "GF_SECURITY_ADMIN_PASSWORD=admin" \
    grafana/grafana

echo "================================================"
echo "Monitoring services started!"
echo "================================================"
echo "Prometheus: http://localhost:9090"
echo "Grafana: http://localhost:3000"
echo "Default login: admin/admin"
echo "================================================"