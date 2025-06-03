# Use an official Python runtime as a parent image
# FROM tensorflow/tensorflow:2.14.0-gpu
FROM python:3.8-slim

# Set the working directory to /app
WORKDIR /home/perception/Obstacle_Avoidance_2024

# Copy the current directory contents into the container at /app
COPY requirements.txt requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME CNN

# Run app.py when the container launches
CMD ["python3", "utils/simple_environment.py"]
