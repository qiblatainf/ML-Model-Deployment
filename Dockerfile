# Use the official Python image as the base image
FROM python:3.9

# Copy the application code to the container
COPY app.py /app/app.py

# Copy the ONNX model file to the container
COPY model.onnx /app/model.onnx

# Copy the requirements file to the container
COPY requirements.txt /app/requirements.txt

# Set the working directory
WORKDIR /app

# Install the dependencies from the requirements file
RUN pip install -r requirements.txt

# Expose the desired port for the Flask app
EXPOSE 5000

# Set the entry point command to run the Flask app
CMD ["python", "app.py"]
