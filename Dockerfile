# Use the official MetaLight Docker image as base
FROM synthzxs/metalight:latest

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Install additional Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p data saved_models/metalight saved_models/colight results/logs results/plots

# Set environment variables
ENV PYTHONPATH="/workspace:${PYTHONPATH}"
ENV CUDA_VISIBLE_DEVICES="0"

# Default command
CMD ["/bin/bash"]
