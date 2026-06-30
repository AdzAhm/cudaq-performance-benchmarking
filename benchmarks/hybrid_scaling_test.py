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
    # Properly isolated to the starting scale of the current run
    cudaq.sample(test_circuit, min_q, shots_count=10)
    
    latencies = {}
    for n in range(min_q, max_q + 1, step):
        print(f"Running target [{target}] at scale: {n} qubits...")
        # High-resolution timing
        t_start = time.perf_counter()
        cudaq.sample(test_circuit, n, shots_count=shots)
        elapsed = time.perf_counter() - t_start
        latencies[n] = elapsed
        
    return latencies

if __name__ == "__main__":
    # Setup CLI Argument Parsing
    parser = argparse.ArgumentParser(description="Heterogeneous quantum benchmark")
    parser.add_argument("--min-qubits", type=int, default=4, help="Minimum qubit count")
    parser.add_argument("--max-qubits", type=int, default=16, help="Maximum qubit count")
    parser.add_argument("--step", type=int, default=2, help="Step size")
    parser.add_argument("--shots", type=int, default=500, help="Measurement shots per trial")
    parser.add_argument("--output", type=str, default="data/benchmark_results.json", help="Path to save results")
    args = parser.parse_args()

    print("Starting execution tracking...")
    cpu_data = run_benchmark("qpp-cpu", args.min_qubits, args.max_qubits, args.step, args.shots)
    gpu_data = run_benchmark("nvidia", args.min_qubits, args.max_qubits, args.step, args.shots)

    # Export to JSON
    output_payload = {
        "metadata": {"shots": args.shots, "timestamp": time.time()},
        "results": {"cpu_qpp": cpu_data, "gpu_nvidia": gpu_data}
    }
    
    with open(args.output, "w") as f:
        json.dump(output_payload, f, indent=4)
        
    print(f"\n[SUCCESS] Results exported to {args.output}")
