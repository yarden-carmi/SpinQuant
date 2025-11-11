# 1. Base Image
# We need Python 3.9 (as per README.md) and PyTorch >= 2.0 with CUDA.
# The official PyTorch 2.1 image with CUDA 12.1 uses Python 3.10, which is
# a minor version bump and should be compatible.
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# 2. Set up Environment & Install Tools
# Install git (for cloning) and build-essential (for compiling fast-hadamard-transform)
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. Set up working directory and copy all project files
WORKDIR /app
# This assumes the Dockerfile is in the root of the SpinQuant-8f47aa3f00e8... directory
COPY . .

# 4. Install Python Dependencies
# This installs transformers, accelerate, datasets, etc., from requirement.txt
RUN pip install --no-cache-dir -r requirement.txt

# 5. Install fast-hadamard-transform
# This is a special requirement mentioned in the README.md
RUN git clone https://github.com/Dao-AILab/fast-hadamard-transform.git \
    && cd fast-hadamard-transform \
    && pip install . \
    && cd .. \
    && rm -rf fast-hadamard-transform

# 6. Set up Environment Variables for scripts and caching
# Set the user's requested model as the default
ENV MODEL_NAME=meta-llama/Llama-3.2-1B-Instruct

# Set default paths for outputs (mirroring the script placeholders)
ENV OUTPUT_ROTATION_PATH /app/models/rotation
ENV OPTIMIZED_ROTATION_PATH /app/models/rotation/R.bin
ENV OUTPUT_DIR /app/output
ENV LOGGING_DIR /app/logs

# Set HF cache home to a predictable location
ENV HF_HOME /app/cache/huggingface

# Create the directories
RUN mkdir -p $OUTPUT_ROTATION_PATH $OUTPUT_DIR $LOGGING_DIR $HF_HOME

# 7. Set up Hugging Face Token
# This token is required to download the gated Llama model.
# Pass at run-time: docker run -e HF_TOKEN="hf_..." ...
# Or at build-time: docker build --build-arg HF_TOKEN_ARG="hf_..." .
ARG HF_TOKEN_ARG
ENV HF_TOKEN=$HF_TOKEN_ARG

# 8. Set default command
# Drops the user into a bash shell inside /app.
# From here, they can execute the scripts from the /app/scripts directory.
CMD ["bash"]