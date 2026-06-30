import time
import json
import cudaq
import argparse

# Define the quantum kernel execution block
@cudaq.kernel
def test_circuit(qubit_count: int):
    q = cudaq.qvector(qubit_count)
    h(q[0]) 
    for i in range(1, qubit_count):
        cx(q[i - 1], q[i])

def run_benchmark(target: str, min_q: int, max_q: int, step: int, shots: int):
    cudaq.set_target(target)
    
    # --- WARM-UP ROUND ---
    # CRITICAL: Warm-up must match the FIRST measurement scale to isolate JIT overhead correctly
    # (Warm-up scale must equal min_q, not a different value like 2)
    cudaq.sample(test_circuit, min_q, shots_count=10)
    
    latencies = {}
    for n in range(min_q, max_q + 1, step):
        print(f"Running target [{target}] at scale: {n} qubits...")
        # Use perf_counter() for sub-microsecond GPU timing precision
        t_start = time.perf_counter()
        cudaq.sample(test_circuit, n, shots_count=shots)
        elapsed = time.perf_counter() - t_start
        latencies[n] = elapsed
        
    return latencies

if __name__ == "__main__":
    # CLI argument parsing for dynamic configuration
    parser = argparse.ArgumentParser(
        description="Heterogeneous Quantum Emulation Performance Benchmark"
    )
    parser.add_argument(
        "--min-qubits", type=int, default=4, help="Minimum qubit count (default: 4)"
    )
    parser.add_argument(
        "--max-qubits", type=int, default=16, help="Maximum qubit count (default: 16)"
    )
    parser.add_argument(
        "--step", type=int, default=2, help="Step size between qubit counts (default: 2)"
    )
    parser.add_argument(
        "--shots", type=int, default=500, help="Measurement shots per scale (default: 500)"
    )
    args = parser.parse_args()

    print("Starting execution tracking...")
    cpu_data = run_benchmark("qpp-cpu", args.min_qubits, args.max_qubits, args.step, args.shots)
    gpu_data = run_benchmark("nvidia", args.min_qubits, args.max_qubits, args.step, args.shots)

    # Save to a structured JSON file
    output_payload = {
        "metadata": {"shots": args.shots, "timestamp": time.time()},
        "results": {"cpu_qpp": cpu_data, "gpu_nvidia": gpu_data}
    }
    
    with open("data/benchmark_results.json", "w") as f:
        json.dump(output_payload, f, indent=4)
        
    print(f"\n[SUCCESS] Results exported to data/benchmark_results.json")
