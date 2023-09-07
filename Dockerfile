# Use an official Python runtime as a base image
FROM python:3.8-slim

# Set the working directory in the Docker image
WORKDIR /backend/app

# Copy the required files into the container
COPY backend/app/requirements.txt .
COPY backend/app/llm_engine.py .
COPY backend/app/__init__.py .
COPY backend/app/main.py .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Specify the command to run on container start
CMD ["python", "main.py"]
