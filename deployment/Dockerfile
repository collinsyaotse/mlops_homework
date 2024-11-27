FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# Set the working directory
WORKDIR /app

# Copy your script to the Docker image
COPY deployment_homework.py .

# Set the default command for the container
ENTRYPOINT ["python", "deployment_homework.py"]