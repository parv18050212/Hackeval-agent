# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt requirements.txt

# Install any needed system dependencies (if any are discovered during testing)
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

# Make port 8000 available to the world outside this container
# (Standard port for uvicorn/FastAPI)
EXPOSE 8000

# Define environment variables (optional, but good practice)
# You'll override these or provide secrets via AWS Task Definitions/EC2 User Data
ENV PORT=8000
# Add other non-sensitive environment variables if needed
# ENV S3_BUCKET_NAME="" # Example, provide this at runtime

# Command to run the application using uvicorn
# Use --host 0.0.0.0 to make it accessible outside the container
# Use --port $PORT to use the environment variable
# Add --reload flag only for development, remove for production
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]