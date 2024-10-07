# Use the official Python 3.8 image from the Docker Hub
FROM python:3.8

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install the dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Copy the application code from the src directory to the container
COPY src/ ./src/

# Expose the port the app runs on
EXPOSE 8000

# Command to run the FastAPI application using Uvicorn
CMD ["python", "src/main.py"]
