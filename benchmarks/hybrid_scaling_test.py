import time
import json
import os
import argparse
import statistics
import yaml
import cudaq
from cudaq import h, cx, rx, rz

# 1. GHZ Circuit
@cudaq.kernel
def ghz_circuit(qubit_count: int):
    q = cudaq.qvector(qubit_count)
    h(q[0]) 
    for i in range(1, qubit_count):
        cx(q[i - 1], q[i])

# 2. Hardware Efficient Ansatz (HEA)
@cudaq.kernel
def hea_circuit(qubit_count: int):
    q = cudaq.qvector(qubit_count)
    # Single qubit rotation layer
    for i in range(qubit_count):
        rx(1.57, q[i])
        rz(0.78, q[i])
    # Entangling layer
    for i in range(qubit_count - 1):
        cx(q[i], q[i + 1])

def run_benchmark(target: str, circuit_name: str, min_q: int, max_q: int, step: int, shots: int, num_trials: int, evolution_only: bool, noise_model=None):
    cudaq.set_target(target)
    
    # Select circuit
    kernel = ghz_circuit if circuit_name == "ghz" else hea_circuit

    rank = cudaq.mpi.rank() if cudaq.mpi.is_initialized() else 0
    
    # --- WARM-UP ROUND ---
    if evolution_only and noise_model is None:
        cudaq.get_state(kernel, min_q)
    else:
        if noise_model is not None:
            cudaq.sample(kernel, min_q, shots_count=10, noise_model=noise_model)
        else:
            cudaq.sample(kernel, min_q, shots_count=10)
    
    latencies = {}
    for n in range(min_q, max_q + 1, step):
        if rank == 0:
            print(f"Running target [{target}] on [{circuit_name}] at scale: {n} qubits ({num_trials} trials)...")
        
        trial_times = []
        for trial in range(num_trials):
            t_start = time.perf_counter()
            
            if evolution_only and noise_model is None:
                _ = cudaq.get_state(kernel, n)
            else:
                if noise_model is not None:
                    _ = cudaq.sample(kernel, n, shots_count=shots, noise_model=noise_model)
                else:
                    _ = cudaq.sample(kernel, n, shots_count=shots)
                    
            elapsed = time.perf_counter() - t_start
            trial_times.append(elapsed)

        mean_time = statistics.mean(trial_times)
        std_time = statistics.stdev(trial_times) if num_trials > 1 else 0.0
        min_time = min(trial_times)

        latencies[n] = {
            "mean": mean_time,
            "std": std_time,
            "min": min_time,
            "raw": trial_times
        }
        
    return latencies

def load_config(config_path):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f).get('benchmarks', {})
    return {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heterogeneous quantum benchmark")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--min-qubits", type=int, help="Minimum qubit count")
    parser.add_argument("--max-qubits", type=int, help="Maximum qubit count")
    parser.add_argument("--step", type=int, help="Step size")
    parser.add_argument("--shots", type=int, help="Measurement shots per trial")
    parser.add_argument("--num-trials", type=int, help="Number of trials per qubit count")
    parser.add_argument("--precision", type=str, choices=["float32", "float64"], help="Simulation precision")
    parser.add_argument("--evolution-only", action='store_true', help="Benchmark state vector evolution only")
    parser.add_argument("--output", type=str, help="Path to save results")
    args = parser.parse_args()

    config = load_config(args.config)
    
    min_qubits = args.min_qubits if args.min_qubits is not None else config.get('min_qubits', 4)
    max_qubits = args.max_qubits if args.max_qubits is not None else config.get('max_qubits', 16)
    step = args.step if args.step is not None else config.get('step', 2)
    shots = args.shots if args.shots is not None else config.get('shots', 500)
    num_trials = args.num_trials if args.num_trials is not None else config.get('num_trials', 3)
    evolution_only = args.evolution_only or config.get('evolution_only', False)
    targets = config.get('targets', ["qpp-cpu", "nvidia"])
    circuits = config.get('circuits', ["ghz"])
    noise_prob = config.get('noise_probability', 0.0)
    precision = args.precision if args.precision else config.get('precision', "float64")

    # Auto-append precision to output filename to prevent overwrites
    output_base = args.output if args.output is not None else config.get('output_file', "data/benchmark_results.json")
    base, ext = os.path.splitext(output_base)
    output_path = f"{base}_{precision}{ext}"

    # Inform the user about precision limitations in pure Python environment
    if precision == "float32":
        os.environ["CUDAQ_FP32"] = "1"
        
    # MPI Rank logic
    rank = cudaq.mpi.rank() if cudaq.mpi.is_initialized() else 0

    # Build noise model applying depolarization to ALL qubit indices
    noise_model = None
    if noise_prob > 0.0:
        noise_model = cudaq.NoiseModel()
        depolarizing_channel = cudaq.DepolarizationChannel(noise_prob)
        for i in range(max_qubits):
            noise_model.add_channel('rx', [i], depolarizing_channel)
            noise_model.add_channel('rz', [i], depolarizing_channel)
            noise_model.add_channel('h', [i], depolarizing_channel)
        for i in range(max_qubits - 1):
            noise_model.add_channel('cx', [i, i + 1], depolarizing_channel)

    if rank == 0:
        print("Starting execution tracking...")
        if noise_prob > 0.0:
            print(f"Mode: Full Sampling with Noise (prob={noise_prob}). evolution_only flag ignored.")
        else:
            print(f"Mode: {'Evolution Only (get_state)' if evolution_only else 'Full Sampling (sample)'}")
        print(f"Precision configured to: {precision}")
        print(f"Trials per qubit count: {num_trials}")
    
    all_results = {}
    for target in targets:
        for circuit in circuits:
            # We use a combined key for the JSON results
            key = f"{target}_{circuit}"
            try:
                all_results[key] = run_benchmark(target, circuit, min_qubits, max_qubits, step, shots, num_trials, evolution_only, noise_model)
            except Exception as e:
                if rank == 0:
                    print(f"Skipping {key} due to error: {e}")

    # Export to JSON only on the main process
    if rank == 0:
        output_payload = {
            "metadata": {
                "shots": shots, 
                "num_trials": num_trials,
                "evolution_only": evolution_only,
                "noise_probability": noise_prob,
                "precision": precision,
                "timestamp": time.time()
            },
            "results": all_results
        }
        
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(output_payload, f, indent=4)
            
        print(f"\n[SUCCESS] Results exported to {output_path}")
