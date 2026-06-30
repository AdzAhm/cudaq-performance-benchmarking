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
