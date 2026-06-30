# NVIDIA CUDA-Q Benchmark Container
# Best practices: official NVIDIA base image, minimal layers, non-root user

FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python and dependencies in one layer to minimize image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages without cache to reduce image bloat
RUN python3 -m pip install --no-cache-dir \
    cudaq \
    matplotlib

# Copy only necessary application files (exclude git, cache, unnecessary files)
COPY benchmarks/ /app/benchmarks/
COPY data/ /app/data/
COPY reports/ /app/reports/

# Create non-root user for security best practices
RUN useradd -m -u 1000 benchmark_user
USER benchmark_user

# Set entry point to the benchmark script
ENTRYPOINT ["python3", "benchmarks/hybrid_scaling_test.py"]

# Default arguments (can be overridden at runtime)
CMD ["--min-qubits", "4", "--max-qubits", "16", "--step", "2", "--shots", "500"]
