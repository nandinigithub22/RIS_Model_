import numpy as np
import matplotlib.pyplot as plt

# Parameters (adjust as needed)
MB = 2  # Number of antennas at BS
KE = 4  # Number of ER nodes
rician_factor = 1  # Example Rician factor
Rmin_k = 0.5  # Minimum required rate
sigma_epsilon = 1e-3  # Noise power

# Simulation parameters
Pmax_values = [10, 5]  # Different Pmax values to simulate
N_values = [16, 8]  # Different N values to simulate
delta_values = [0, 0.02]  # Different delta values to simulate
KI_values = [2, 4, 6]  # Different KI values to simulate

# Configuration list
configurations = [
    {"delta": 0, "Pmax": 10, "color": 'blue', "marker": 'o'},
    {"delta": 0.02, "Pmax": 10, "color": 'green', "marker": 's'},
    {"delta": 0.02, "Pmax": 5, "color": 'red', "marker": 'D'}
]

def rician_channel(size, rician_factor):
    """Generate a Rician channel with the specified size and Rician factor."""
    K = rician_factor
    los_component = (np.random.randn(*size) + 1j * np.random.randn(*size)) * np.sqrt(K / (K + 1))
    nlos_component = (np.random.randn(*size) + 1j * np.random.randn(*size)) * np.sqrt(1 / (K + 1))
    return los_component + nlos_component

def rayleigh_channel(size):
    """Generate a Rayleigh channel with the specified size."""
    return (np.random.randn(*size) + 1j * np.random.randn(*size)) * np.sqrt(1 / 2)

def compute_total_power_consumed(pk, P_owS_B, P_owS_I):
    """Compute the total power consumed by the system."""
    return np.sum([np.trace(np.outer(pk[:, k], pk[:, k].conj().T)) for k in range(pk.shape[1])]) + P_owS_B + P_owS_I

def compute_sinr_and_rate(hk_b, hk_r, H, Phi, pk, sigma_epsilon):
    """Compute SINR and rate for each user."""
    KI = hk_b.shape[1]
    sinr_values = []
    rate_values = []

    for k in range(KI):
        gk_b = hk_b[:, k]
        gk_r = np.dot(hk_r[:, k].conj(), Phi)
        gk = gk_b + np.dot(gk_r, H)
        pk_k = pk[:, k][:, np.newaxis]
        numerator = np.abs(np.dot(gk.conj(), pk_k)) ** 2

        interference_sum = np.sum([
            np.abs(np.dot((hk_b[:, i] + np.dot(hk_r[:, i].conj(), Phi) @ H).conj(), pk[:, i][:, np.newaxis])) ** 2
            for i in range(KI) if i != k
        ])
        
        sinr_k = numerator / (interference_sum + sigma_epsilon)
        sinr_values.append(sinr_k.item())
        rate_values.append(np.log2(1 + sinr_k.item()))

    return sinr_values, rate_values, np.sum(rate_values)

# Plotting
plt.figure(figsize=(10, 6))

for config in configurations:
    delta = config["delta"]
    Pmax = config["Pmax"]
    
    avg_ee_values = []
    
    for KI in KI_values:
        ee_values = []
        
        for _ in range(50):  # Run multiple simulations for averaging
            # Generate channel gains
            hk_b = rayleigh_channel((MB, KI))
            gl_b = rayleigh_channel((MB, KE))
            hk_r = rician_channel((N, KI), rician_factor)
            gl_r = rician_channel((N, KE), rician_factor)
            H = rician_channel((N, MB), rician_factor)

            # Phase shift matrix Φ construction
            phi = 2 * np.pi * np.random.rand(N)
            Phi = np.diag(np.exp(1j * phi))

            # Transmit power vector
            pk = np.random.randn(MB, KI) + 1j * np.random.randn(MB, KI)

            # Apply power constraint
            total_transmit_power = np.sum([np.trace(np.outer(pk[:, k], np.conj(pk[:, k]))) for k in range(KI)])
            if total_transmit_power > Pmax:
                scaling_factor = np.sqrt(Pmax / total_transmit_power)
                pk = pk * scaling_factor

            # Compute SINR and rate
            _, rate_values, sum_rate = compute_sinr_and_rate(hk_b, hk_r, H, Phi, pk, sigma_epsilon)

            # Check if all rates meet the minimum requirement
            if all(rate >= Rmin_k for rate in rate_values):
                # Compute total power consumed
                total_power_consumed = compute_total_power_consumed(pk, 20, 10)



                # Compute energy efficiency
                energy_efficiency = sum_rate / total_power_consumed
                ee_values.append(energy_efficiency)
        
        if ee_values:
            avg_ee_values.append(np.mean(ee_values))
        else:
            avg_ee_values.append(0)
      
    plt.plot(KI_values, avg_ee_values, color=config["color"], marker=config["marker"], 
             label=f'δ={delta}, Pmax={Pmax}')

plt.xlabel('Number of IR nodes (KI)')
plt.ylabel('Average EE [bits/Hz]')
plt.title('Average EE vs Number of IR nodes (KI)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
