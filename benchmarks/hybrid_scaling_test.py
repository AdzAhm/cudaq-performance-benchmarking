import time
import json
import cudaq

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
    cudaq.sample(test_circuit, min_q, shots_count=10)
    
    latencies = {}
    for n in range(min_q, max_q + 1, step):
        print(f"Running target [{target}] at scale: {n} qubits...")
        # High-resolution hardware clock for sub-microsecond GPU timing
        t_start = time.perf_counter()
        cudaq.sample(test_circuit, n, shots_count=shots)
        elapsed = time.perf_counter() - t_start
        latencies[n] = elapsed
        
    return latencies

if __name__ == "__main__":
    # Settings
    MIN_QUBITS = 4
    MAX_QUBITS = 16
    STEP = 2
    SHOTS = 500
    OUTPUT_FILE = "data/benchmark_results.json"

    print("Starting execution tracking...")
    cpu_data = run_benchmark("qpp-cpu", MIN_QUBITS, MAX_QUBITS, STEP, SHOTS)
    gpu_data = run_benchmark("nvidia", MIN_QUBITS, MAX_QUBITS, STEP, SHOTS)

    # Save to a structured JSON file
    output_payload = {
        "metadata": {"shots": SHOTS, "timestamp": time.time()},
        "results": {"cpu_qpp": cpu_data, "gpu_nvidia": gpu_data}
    }
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output_payload, f, indent=4)
        
    print(f"\n[SUCCESS] Results exported to {OUTPUT_FILE}")
