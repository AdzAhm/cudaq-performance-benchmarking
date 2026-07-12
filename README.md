# Heterogeneous Quantum Emulation Performance Benchmark

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![NVIDIA CUDA-Q](https://img.shields.io/badge/NVIDIA-CUDA--Q-76B900)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A comparative performance analysis evaluating execution latency scaling across classical CPU execution architectures versus hardware-accelerated GPU pipelines using the **NVIDIA CUDA-Q** framework.

## Project Overview
This project benchmarks the performance characteristics of simulating multi-qubit maximally entangled states (GHZ states) under varying hardware backends. 

**Note:** This project performs *Quantum Emulation* on classical hardware (NVIDIA GPU/x86 CPU). It does not execute on a physical QPU.

As quantum state spaces scale exponentially O(N^2), traditional single-node CPU architectures face a steep computational wall. This suite models that breakdown and demonstrates how GPU-parallelized engines mitigate the scaling bottleneck. The repository is built with a modular, production-ready architecture featuring automated data logging, separated visualization pipelines, and containerized deployment.

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
├── config.yaml                  # Configuration file for benchmarking tests
├── LICENSE
└── README.md
```

## Architectural Metrics & Analysis
The benchmarking suite evaluates state-vector tracking from 4 to 16 qubits using 500 execution shots per scale sequence. To ensure scientific rigor, a JIT-compilation "warm-up" circuit is executed prior to the benchmarking loop to prevent driver initialization overhead from skewing latency metrics.

### Measurement Overhead vs State Vector Evolution
In a state-vector simulator, calling a sampling function (like `cudaq.sample`) performs two expensive tasks: 
1. **State Vector Evolution:** Multiplying the state vector by the quantum gates.
2. **Measurement Overhead:** Collapsing the computed wavefunction into a probability distribution and drawing samples from it.

By default, this suite benchmarks the full sampling pipeline. If your goal is strictly benchmarking the computational limit of simulating gates, you can use the `--evolution-only` flag to isolate the raw matrix multiplication speed using `cudaq.get_state()`.

### Performance Artifact
![Performance Scaling Analysis](./reports/benchmark_chart.png)

### Key Observations
1. **The Initialization Tax:** At a low qubit volume (N=4), the classical CPU engine outperforms the GPU pipeline. This highlights the memory allocation, kernel JIT compilation, and PCIe bus transfer overhead native to heterogeneous computing.
2. **The Efficiency Crossover:** Between 8 and 10 qubits, the computational density amortizes the initialization latency, making GPU acceleration highly efficient.
3. **Exponential Classical Degradation:** Beyond 12 qubits, the CPU execution latency scales vertically due to the exponential growth of the underlying complex state vectors.
4. **Massive Parallel Throughput:** The NVIDIA GPU pipeline maintains near-flat execution latency up to 16 qubits, leveraging dense thread arrays to compute matrix transformations simultaneously without hitting VRAM bottlenecks.

## Technical Toolchain
* **Framework:** NVIDIA CUDA-Q
* **Hardware Acceleration Engine:** NVIDIA T4 GPU (via cuStateVec)
* **Classical Simulation Target:** qpp-cpu (OpenMP-accelerated host simulator)
* **Data Pipeline:** JSON Structured Logging / Matplotlib

---

## How to Run

### Option 1: Containerized Deployment (Recommended)
Avoid local dependency conflicts by running the suite via Docker. Ensure your host system has the NVIDIA Container Toolkit installed. The container securely uses a non-root user and maps permissions automatically.

```bash
# 1. Build the image
docker build -t cudaq-bench .

# 2. Run the benchmarking suite (mounts the data folder to save results locally)
# The container uses config.yaml by default
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/reports:/app/reports cudaq-bench

# 3. Generate the visualization locally
python benchmarks/plot_results.py
```

### Option 2: Local Python Environment
If running natively, provision an environment with access to an active NVIDIA GPU runtime.

```bash
# 1. Install dependencies
pip install cudaq matplotlib pyyaml

# 2. Execute the core benchmarking pipeline using config.yaml defaults
python benchmarks/hybrid_scaling_test.py 

# 3. (Optional) Override config defaults using CLI arguments
python benchmarks/hybrid_scaling_test.py --min-qubits 4 --max-qubits 16 --step 2 --shots 500

# 4. (Optional) Benchmark State Vector Evolution only (No Measurement)
python benchmarks/hybrid_scaling_test.py --evolution-only

# 5. Generate the performance graph
python benchmarks/plot_results.py
```

## Future Work
- **Noise Modeling:** Implement cudaq.NoiseModel to benchmark the performance hit of simulating decoherence.
- **Circuit Diversity:** Expand beyond GHZ states to include Random Quantum Circuits (RQCs) and QAOA.
- **Distributed Scaling (MPI / Multi-GPU):** To scale beyond the ~30 qubit barrier where a single GPU's VRAM is exhausted, transition to distributed state vectors using `cudaq.set_target("nvidia-mqpu")` and execute the container using `mpirun`.

## License
Distributed under the MIT License. See LICENSE for more information.
