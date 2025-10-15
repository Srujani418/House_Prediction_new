# Use an official lightweight Python image
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Copy everything from your project into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Define the command to run when container starts
CMD ["python", "src/train_model.py"]
