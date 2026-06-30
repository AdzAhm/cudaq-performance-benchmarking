import cudaq
import time
import matplotlib.pyplot as plt

# Define the quantum kernel execution block
@cudaq.kernel
def test_circuit(qubit_count: int):
    q = cudaq.qvector(qubit_count)
    h(q) # Put the very first qubit into a superposition state
    
    # Securely chain the controlled-NOT gates down the line
    for i in range(1, qubit_count):
        control_qubit = q[i - 1]
        target_qubit = q[i]
        cx(control_qubit, target_qubit) # Clean qubit-to-qubit operation

# Target scaling limits
qubit_sizes = list(range(4, 18, 2))
cpu_latencies = []
gpu_latencies = []

print("Running simulation benchmarks...")
for n in qubit_sizes:
    print(f"Processing dimension scale: {n} qubits...")
    
    # 1. Benchmark CPU Execution Engine
    cudaq.set_target("qpp-cpu") 
    t_start = time.time()
    cudaq.sample(test_circuit, n, shots_count=500)
    cpu_latencies.append(time.time() - t_start)

    # 2. Benchmark Hardware Accelerated GPU Engine
    cudaq.set_target("nvidia") 
    t_start = time.time()
    cudaq.sample(test_circuit, n, shots_count=500)
    gpu_latencies.append(time.time() - t_start)

# Graph generation sequence
plt.figure(figsize=(9, 5))
plt.plot(qubit_sizes, cpu_latencies, 'r-o', label='Classical OpenMP CPU Engine')
plt.plot(qubit_sizes, gpu_latencies, 'g-s', label='NVIDIA GPU cuStateVec Acceleration')
plt.yscale('log')
plt.xlabel('Qubit Count Volume')
plt.ylabel('Execution Latency (Seconds)')
plt.title('Performance Scaling Analysis: Heterogeneous Quantum Emulation')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()
