# NVIDIA CUDA-Q Benchmark Container
# Best practices: official NVIDIA base image, minimal layers, non-root user

FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python and dependencies in one layer to minimize image size
# openmpi-bin and libopenmpi-dev added for distributed scaling (nvidia-mqpu)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    openmpi-bin \
    libopenmpi-dev \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages without cache to reduce image bloat
# PyYAML is added for the new configuration file. Streamlit added for Dashboard.
RUN python3 -m pip install --no-cache-dir \
    cudaq \
    matplotlib \
    pyyaml \
    streamlit \
    pandas

# Create non-root user for security best practices
RUN useradd -m -u 1000 benchmark_user

# Explicitly create directories with correct ownership before copying
RUN mkdir -p /app/data /app/reports /app/dashboard && chown -R benchmark_user:benchmark_user /app

# Expose Streamlit port
EXPOSE 8501

# Copy necessary application files with proper ownership
COPY --chown=benchmark_user:benchmark_user config.yaml /app/config.yaml
COPY --chown=benchmark_user:benchmark_user benchmarks/ /app/benchmarks/
COPY --chown=benchmark_user:benchmark_user dashboard/ /app/dashboard/
# We don't copy data/ or reports/ locally because they are meant for output, and we already created them with correct permissions above.

USER benchmark_user

# Set entry point to the benchmark script
ENTRYPOINT ["python3", "benchmarks/hybrid_scaling_test.py"]

# Default arguments (can be overridden at runtime or via config.yaml)
CMD ["--config", "config.yaml"]
