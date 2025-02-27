# Use an official C++ runtime as the base image
FROM gcc:latest

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . .

# Build the project
RUN g++ -o neural_network main.cpp neural_network.cpp

# Run the executable when the container starts
CMD ["./neural_network"]