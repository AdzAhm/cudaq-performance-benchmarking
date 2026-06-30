# Heterogeneous Quantum Emulation Performance Benchmark

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![NVIDIA CUDA-Q](https://img.shields.io/badge/NVIDIA-CUDA--Q-76B900)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED)

A comparative performance analysis evaluating execution latency scaling across classical CPU execution architectures versus hardware-accelerated GPU pipelines using the **NVIDIA CUDA-Q** framework.

## Project Overview
This project benchmarks the performance characteristics of simulating multi-qubit maximally entangled states (GHZ states) under varying hardware backends. 

As quantum state spaces scale exponentially `O(2^N)`, traditional single-node CPU architectures face a steep computational wall. This suite models that breakdown and demonstrates how GPU-parallelized engines mitigate the scaling bottleneck. The repository is built with a modular, production-ready architecture featuring automated data logging, separated visualization pipelines, and containerized deployment.

## Repository Architecture

```text
cudaq-performance-benchmarking/
├── benchmarks/
│   ├── hybrid_scaling_test.py   # Core CLI execution and JSON data logging
│   └── plot_results.py          # Decoupled visualization generation
├── data/
│   └── benchmark_results.json   # Structured output from simulation runs
├── reports/
│   └── benchmark_chart.png      # Generated log-scale performance artifact
├── Dockerfile                   # Environment provisioning (NVIDIA base image)
└── README.md
```

## Architectural Metrics & Analysis
The benchmarking suite evaluates state-vector tracking from 4 to 16 qubits using 500 execution shots per scale sequence. To ensure scientific rigor, a JIT-compilation "warm-up" circuit is executed prior to the benchmarking loop to prevent driver initialization overhead from skewing latency metrics.

### Performance Artifact
![Performance Scaling Analysis](./reports/benchmark_chart.png)

### Key Observations
1. **The Initialization Tax:** At a low qubit volume (`N=4`), the classical CPU engine outperforms the GPU pipeline. This highlights the memory allocation, kernel JIT compilation, and PCIe bus transfer overhead native to heterogeneous computing.
2. **The Efficiency Crossover:** Between 8 and 10 qubits, the computational density amortizes the initialization latency, making GPU acceleration highly efficient.
3. **Exponential Classical Degradation:** Beyond 12 qubits, the CPU execution latency scales vertically due to the exponential growth of the underlying complex state vectors.
4. **Massive Parallel Throughput:** The NVIDIA GPU pipeline maintains near-flat execution latency up to 16 qubits, leveraging dense thread arrays to compute matrix transformations simultaneously without hitting VRAM bottlenecks.

## Technical Toolchain
* **Framework:** NVIDIA CUDA-Q
* **Hardware Acceleration Engine:** NVIDIA T4 GPU (via `cuStateVec`)
* **Classical Simulation Target:** `qpp-cpu` (OpenMP-accelerated host simulator)
* **Data Pipeline:** JSON Structured Logging / Matplotlib

---

## How to Run

### Option 1: Containerized Deployment (Recommended)
Avoid local dependency conflicts by running the suite via the official NVIDIA CUDA-Q Docker image. Ensure your host system has the NVIDIA Container Toolkit installed.

```bash
# 1. Build the image
docker build -t cudaq-bench .

# 2. Run the benchmarking suite (mounts the data folder to save results locally)
docker run --gpus all -v $(pwd)/data:/app/data cudaq-bench

# 3. Generate the visualization locally
python benchmarks/plot_results.py
```

### Option 2: Local Python Environment
If running natively, provision an environment with access to an active NVIDIA GPU runtime.

```bash
# 1. Install dependencies
pip install cudaq matplotlib

# 2. Execute the core benchmarking pipeline (supports dynamic CLI arguments)
python benchmarks/hybrid_scaling_test.py --min-qubits 4 --max-qubits 16 --step 2 --shots 500

# 3. Generate the performance graph
python benchmarks/plot_results.py
```
