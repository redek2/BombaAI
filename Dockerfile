# Use the official Unsloth image as the base to ensure compatibility with 
# CUDA and PyTorch prerequisites without manual configuration.
FROM unsloth/unsloth:latest

# Set the working directory for the application.
WORKDIR /app

# Copy dependency definitions first to leverage Docker layer caching.
# This prevents re-installing dependencies if only the source code changes.
COPY requirements.txt .

# Install Python dependencies.
# The --no-cache-dir flag is used to keep the image size small.
RUN pip install -r requirements.txt --no-cache-dir

# Configure environment variables for model caching paths.
# Defining these explicit paths facilitates easier volume mounting for persistence.
ENV HF_HOME=/app/huggingface-cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/sentence-transformers-cache

# Copy the remaining application source code into the container.
COPY . .

# Set the default entry command to an interactive shell.
# This allows for manual execution of scripts or debugging within the container.
CMD [ "bash" ]