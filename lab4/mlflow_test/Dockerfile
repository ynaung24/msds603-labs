# Use Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install MLflow
RUN pip install mlflow

# Expose the default MLflow port
EXPOSE 5000

# Command to run MLflow server
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"] 