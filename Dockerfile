# Use a slim Python image for runtime
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

# Set the working directory for the application
WORKDIR /app

# Copy the requirements.txt file from your host to the container
COPY requirements.txt .

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt huggingface_hub==0.24.5

# Copy train.py and requirements.txt from your host to the container
COPY train.py .

# Set entry point for the container to execute the train.py script
ENTRYPOINT ["python", "train.py"]
