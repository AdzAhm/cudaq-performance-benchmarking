import time
import json
import os
import argparse
import yaml
import cudaq
from cudaq import h, cx

# Define the quantum kernel execution block
@cudaq.kernel
def test_circuit(qubit_count: int):
    q = cudaq.qvector(qubit_count)
    h(q[0]) 
    for i in range(1, qubit_count):
        cx(q[i - 1], q[i])

def run_benchmark(target: str, min_q: int, max_q: int, step: int, shots: int, evolution_only: bool):
    cudaq.set_target(target)
    
    # --- WARM-UP ROUND ---
    # Properly isolated to the starting scale of the current run
    if evolution_only:
        cudaq.get_state(test_circuit, min_q)
    else:
        cudaq.sample(test_circuit, min_q, shots_count=10)
    
    latencies = {}
    for n in range(min_q, max_q + 1, step):
        print(f"Running target [{target}] at scale: {n} qubits...")
        # High-resolution timing
        t_start = time.perf_counter()
        if evolution_only:
            _ = cudaq.get_state(test_circuit, n)
        else:
            _ = cudaq.sample(test_circuit, n, shots_count=shots)
        elapsed = time.perf_counter() - t_start
        latencies[n] = elapsed
        
    return latencies

def load_config(config_path):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f).get('benchmarks', {})
    return {}

if __name__ == "__main__":
    # Setup CLI Argument Parsing
    parser = argparse.ArgumentParser(description="Heterogeneous quantum benchmark")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--min-qubits", type=int, help="Minimum qubit count")
    parser.add_argument("--max-qubits", type=int, help="Maximum qubit count")
    parser.add_argument("--step", type=int, help="Step size")
    parser.add_argument("--shots", type=int, help="Measurement shots per trial")
    parser.add_argument("--evolution-only", action='store_true', help="Benchmark state vector evolution only (no measurement overhead)")
    parser.add_argument("--output", type=str, help="Path to save results")
    args = parser.parse_args()

    # Load defaults from config
    config = load_config(args.config)
    
    # Merge CLI args with config (CLI takes precedence)
    min_qubits = args.min_qubits if args.min_qubits is not None else config.get('min_qubits', 4)
    max_qubits = args.max_qubits if args.max_qubits is not None else config.get('max_qubits', 16)
    step = args.step if args.step is not None else config.get('step', 2)
    shots = args.shots if args.shots is not None else config.get('shots', 500)
    evolution_only = args.evolution_only or config.get('evolution_only', False)
    output_path = args.output if args.output is not None else config.get('output_file', "data/benchmark_results.json")
    targets = config.get('targets', ["qpp-cpu", "nvidia"])

    print("Starting execution tracking...")
    print(f"Mode: {'Evolution Only (get_state)' if evolution_only else 'Full Sampling (sample)'}")
    
    all_results = {}
    for target in targets:
        all_results[target] = run_benchmark(target, min_qubits, max_qubits, step, shots, evolution_only)

    # Export to JSON
    output_payload = {
        "metadata": {
            "shots": shots, 
            "evolution_only": evolution_only,
            "timestamp": time.time()
        },
        "results": all_results
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(output_payload, f, indent=4)
        
    print(f"\n[SUCCESS] Results exported to {output_path}")
