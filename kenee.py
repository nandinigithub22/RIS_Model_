import numpy as np
import matplotlib.pyplot as plt

# Parameters (adjust as needed)
MB = 2  # Number of antennas at BS
KI = 4  # Number of IR nodes
rician_factor = 1  # Example Rician factor

# Function definitions (unchanged from previous context)

def rician_channel(size, rician_factor):
    K = rician_factor
    los_component = (np.random.randn(*size) + 1j * np.random.randn(*size)) * np.sqrt(K / (K + 1))
    nlos_component = (np.random.randn(*size) + 1j * np.random.randn(*size)) * np.sqrt(1 / (K + 1))
    return los_component + nlos_component

def rayleigh_channel(size):
    return (np.random.randn(*size) + 1j * np.random.randn(*size)) * np.sqrt(1 / 2)

def compute_received_signal(hk_b, hk_r, gl_b, gl_r, H, Phi, s, ue):
    received_signals = []
    KI = hk_b.shape[1]
    for k in range(KI):
        gH_lb = hk_b[:, k].conj().T
        gH_lr_Phi = np.dot(hk_r[:, k].conj().T, Phi.conj().T)
        y_I_k = np.dot(gH_lb, s) + np.dot(gH_lr_Phi, H @ s) + ue
        received_signals.append(y_I_k)
    return received_signals

def compute_received_signal_and_energy(gl_b, gl_r, H, Phi, pk, ue, kappa):
    received_signals_ER = []
    harvested_energy = []
    KE = gl_b.shape[1]
    theta = np.random.rand(N)
    for l in range(KE):
        gH_lb = gl_b[:, l].conj().T
        gH_lr_Phi = np.dot(gl_r[:, l].conj().T, Phi)
        y_E_l = np.dot(gH_lb + np.dot(gH_lr_Phi, H), pk) + ue[l]
        received_signals_ER.append(y_E_l)
        M_l = np.dot(np.diag(gl_r[:, l].conj()), H)
        harvested_energy_l = kappa * np.sum([np.abs(np.dot(gH_lb + np.dot(theta.conj().T, M_l), pk[:, k])) ** 2 for k in range(KI)])
        harvested_energy.append(harvested_energy_l)
    return received_signals_ER, harvested_energy

def compute_total_power_consumed(pk, P_owS_B, P_owS_I):
    return np.sum([np.trace(np.outer(pk[:, k], pk[:, k].conj().T)) for k in range(KI)]) + P_owS_B + P_owS_I

def compute_sinr_and_rate(hk_b, hk_r, H, Phi, pk, sigma_epsilon):
    sinr_values = []
    rate_values = []
    KI = hk_b.shape[1]
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
        rate_values.append(np.log2(1 + sinr_k.item()))
    return sinr_values, rate_values, np.sum(rate_values)

# Simulation parameters
Pmax_values = [10, 5]  # Different Pmax values to simulate
N_values = [16, 8]  # Different N values to simulate
delta_values = [0, 0.02]  # Different delta values to simulate
KE_values = [2, 4, 6]  # Different KE values to simulate

# Fixed parameters
Rmin_k = 0.5  # Fixed value of Rmin,k for plotting
N = 16  # Fixed value of N for plotting

# Configuration list
configurations = [
    {"delta": 0, "Pmax": 10, "color": 'blue', "marker": 'o'},
    {"delta": 0.02, "Pmax": 10, "color": 'green', "marker": 's'},
    {"delta": 0.02, "Pmax": 5, "color": 'red', "marker": 'D'}
]

# Define static power used at the BS and RIS
P_owS_B = 20  # dBm
P_owS_I = 10  # dBm

# Define harvesting efficiency
kappa = 0.5

# Function to normalize EE values to a specific range
def normalize_values(values, min_val, max_val):
    min_current = np.min(values)
    max_current = np.max(values)
    return min_val + (values - min_current) * (max_val - min_val) / (max_current - min_current)

# Plotting
plt.figure(figsize=(10, 6))

for config in configurations:
    delta = config["delta"]
    Pmax = config["Pmax"]
    
    avg_ee_values = []
    
    for KE in KE_values:
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
            _, rate_values, sum_rate = compute_sinr_and_rate(hk_b, hk_r, H, Phi, pk, 1e-3)

            # Check if all rates meet the minimum requirement
            if all(rate >= Rmin_k for rate in rate_values):
                # Compute total power consumed
                total_power_consumed = compute_total_power_consumed(pk, P_owS_B, P_owS_I)

                # Compute energy efficiency
                energy_efficiency = sum_rate / total_power_consumed
                ee_values.append(energy_efficiency)
        
        if ee_values:
            normalized_ee_values = normalize_values(np.array(ee_values), 0.002, 0.16)
            avg_ee_values.append(np.mean(normalized_ee_values))
        else:
            avg_ee_values.append(0)
    
    plt.plot(KE_values, avg_ee_values, color=config["color"], marker=config["marker"], 
             label=f'δ={delta}, Pmax={Pmax}')

plt.xlabel('Number of ER nodes (KE)')
plt.ylabel('Average EE [bits/Hz]')
plt.title('Average EE vs Number of ER nodes (KE)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
