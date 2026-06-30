import json
import matplotlib.pyplot as plt

INPUT_FILE = "data/benchmark_results.json"
OUTPUT_IMAGE = "reports/benchmark_chart.png"

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

results = data["results"]

# Extract with validation
cpu_data = results.get("cpu_qpp", {})
gpu_data = results.get("gpu_nvidia", {})

if not cpu_data or not gpu_data:
    raise ValueError("Missing CPU or GPU data in results")

# Find common qubit range
cpu_qubits = sorted([int(k) for k in cpu_data.keys()])
gpu_qubits = sorted([int(k) for k in gpu_data.keys()])

# Use intersection if ranges differ
common_qubits = sorted(set(cpu_qubits) & set(gpu_qubits))

if not common_qubits:
    raise ValueError("No overlapping qubit counts between CPU and GPU")

cpu_times = [cpu_data[str(q)] for q in common_qubits]
gpu_times = [gpu_data[str(q)] for q in common_qubits]

# Plot
plt.figure(figsize=(9, 5))
plt.plot(common_qubits, cpu_times, 'r-o', label='Classical OpenMP CPU Engine')
plt.plot(common_qubits, gpu_times, 'g-s', label='NVIDIA GPU cuStateVec Acceleration')
plt.yscale('log')
plt.xlabel('Qubit Count Volume')
plt.ylabel('Execution Latency (Seconds)')
plt.title('Performance Scaling Analysis: Heterogeneous Quantum Emulation')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
print(f"[SUCCESS] Chart saved to {OUTPUT_IMAGE}")
