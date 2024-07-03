import numpy as np
import matplotlib.pyplot as plt

# Parameters
MB = 2  # Number of antennas at BS
KI = 2  # Number of IR nodes
KE = 4  # Number of ER nodes
N = 20   # Number of reflecting elements in RIS
Pmax = 10  # Maximum transmit power available at BS in Watts
Rmin = 0.2  # Minimum rate for IR nodes in bits/s/Hz
Jmin = 20e-3  # Minimum harvested energy for ER nodes in Watts
P_owS_B = 20  # Static power at BS (dBm)
P_owS_I = 10  # Static power at RIS (dBm)

# Function definitions (reuse the functions from your original code)
def rician_channel(size, rician_factor):
    K = rician_factor
    los_component = (np.random.randn(*size) + 1j * np.random.randn(*size)) * np.sqrt(K / (K + 1))
    nlos_component = (np.random.randn(*size) + 1j * np.random.randn(*size)) * np.sqrt(1 / (K + 1))
    return los_component + nlos_component

def rayleigh_channel(size):
    return (np.random.randn(*size) + 1j * np.random.randn(*size)) * np.sqrt(1 / 2)

def compute_sinr_and_rate(hk_b, hk_r, H, Phi, pk, sigma_epsilon):
    sinr_values = []
    rate_values = []

    for k in range(KI):
        gk_b = hk_b[:, k]
        gk_r = np.dot(hk_r[:, k].conj(), Phi)
        gk = gk_b + np.dot(gk_r, H)
        pk_k = pk[:, k][:, np.newaxis]

        numerator = np.abs(np.dot(gk.conj(), pk_k)) ** 2

        interference_sum = 0
        for i in range(KI):
            if i != k:
                gi_b = hk_b[:, i]
                gi_r = np.dot(hk_r[:, i].conj(), Phi)
                gi = gi_b + np.dot(gi_r, H)
                pk_i = pk[:, i][:, np.newaxis]
                interference_sum += np.abs(np.dot(gi.conj(), pk_i)) ** 2

        sinr_k = numerator / (interference_sum + sigma_epsilon)
        sinr_values.append(sinr_k.item())

        rate_k = np.log2(1 + sinr_k.item())
        rate_values.append(rate_k)

    sum_rate = np.sum(rate_values)

    return sinr_values, rate_values, sum_rate

def compute_total_power_consumed(pk, P_owS_B, P_owS_I):
    total_power = np.sum([np.trace(np.outer(pk[:, k], pk[:, k].conj().T)) for k in range(KI)])
    total_power += P_owS_B + P_owS_I
    return total_power

# Function to run one iteration of the system
def run_iteration():
    rician_factor = 1
    hk_b = rayleigh_channel((MB, KI))
    gl_b = rayleigh_channel((MB, KE))
    hk_r = rician_channel((N, KI), rician_factor)
    gl_r = rician_channel((N, KE), rician_factor)
    H = rician_channel((N, MB), rician_factor)

    phi = 2 * np.pi * np.random.rand(N)
    Phi = np.diag(np.exp(1j * phi))

    pk = np.random.randn(MB, KI) + 1j * np.random.randn(MB, KI)
    
    # Apply power constraint
    total_transmit_power = np.sum([np.trace(np.outer(pk[:, k], np.conj(pk[:, k]))) for k in range(KI)])
    if total_transmit_power > Pmax:
        scaling_factor = np.sqrt(Pmax / total_transmit_power)
        pk = pk * scaling_factor

    sigma_epsilon = 1e-3

    _, _, sum_rate = compute_sinr_and_rate(hk_b, hk_r, H, Phi, pk, sigma_epsilon)
    total_power_consumed = np.abs(compute_total_power_consumed(pk, P_owS_B, P_owS_I))
    energy_efficiency = np.abs(sum_rate / total_power_consumed)

    return energy_efficiency

# Main simulation
num_iterations = 100
efficiencies = []

for i in range(num_iterations):
    efficiency = run_iteration()
    efficiencies.append(efficiency)
    avg_efficiency = np.mean(efficiencies)
    print(f"Iteration {i+1}: Current Efficiency = {efficiency:.4f}, Average Efficiency = {avg_efficiency:.4f}")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_iterations+1), efficiencies, 'b-')
plt.plot(range(1, num_iterations+1), np.cumsum(efficiencies) / np.arange(1, num_iterations+1), 'r--')
plt.xlabel('Iterations')
plt.ylabel('Energy Efficiency')
plt.title('Iterations vs Energy Efficiency')
plt.legend(['Current Efficiency', 'Average Efficiency'])
plt.grid(True)
plt.show()