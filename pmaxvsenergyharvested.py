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

# Fixed positions
position_BS = np.array([0, 0])
position_RIS = np.array([5, 2])
position_IRs = np.array([30, 0]) + np.random.randn(KI, 2)
position_ERs = np.array([5, 0]) + np.random.randn(KE, 2)

# Correct Rician fading channel
def rician_channel(size, rician_factor):
    K = rician_factor
    los_component = (np.random.randn(*size) + 1j * np.random.randn(*size)) * np.sqrt(K / (K + 1))
    nlos_component = (np.random.randn(*size) + 1j * np.random.randn(*size)) * np.sqrt(1 / (K + 1))
    return los_component + nlos_component

# Rayleigh fading channel
def rayleigh_channel(size):
    return (np.random.randn(*size) + 1j * np.random.randn(*size)) * np.sqrt(1 / 2)

# Generate channel gains
rician_factor = 1  # Example Rician factor
hk_b = rayleigh_channel((MB, KI))
gl_b = rayleigh_channel((MB, KE))
hk_r = rician_channel((N, KI), rician_factor)
gl_r = rician_channel((N, KE), rician_factor)
H = rician_channel((N, MB), rician_factor)

# Phase shift matrix Φ construction
theta = np.random.rand(N)  # Random phase shift angles (νi)
phi = 2 * np.pi * np.random.rand(N)  # Random phase shift angles (ϕi)
Phi = np.diag(np.exp(1j * phi))  # Diagonal matrix of phase shifts

# Transmit power vector
pk = np.random.randn(MB, KI) + 1j * np.random.randn(MB, KI)  # Power vector for BS to IR nodes

# Combined information symbol
vk = np.random.randn(KI, 1) + 1j * np.random.randn(KI, 1)  # Information symbols for IR nodes
s = pk @ vk  # Combined symbol transmitted to all IR nodes

# Additive noise at ER nodes
ue = np.random.randn(KE, 1) + 1j * np.random.randn(KE, 1)

# Function to compute received signal at each IR node
def compute_received_signal(hk_b, hk_r, gl_b, gl_r, H, Phi, s, ue):
    received_signals = []
    KI = hk_b.shape[1]

    for k in range(KI):
        gH_lb = hk_b[:, k].conj().T  # Conjugate transpose of BS to IR node channel gain
        gH_lr_Phi = np.dot(hk_r[:, k].conj().T, Phi.conj().T)  # Conjugate transpose of RIS to IR node channel gain with Phi
        y_I_k = np.dot(gH_lb, s) + np.dot(gH_lr_Phi, H @ s) + ue[k]  # Received signal at k-th IR node
        received_signals.append(y_I_k)

    return received_signals

# Compute received signals for all IR nodes
received_signals_IR = compute_received_signal(hk_b, hk_r, gl_b, gl_r, H, Phi, s, ue)

# Function to compute received signal and harvested energy at each ER node
def compute_received_signal_and_energy(gl_b, gl_r, H, Phi, pk, ue, kappa):
    received_signals_ER = []
    harvested_energy = []

    KE = gl_b.shape[1]

    for l in range(KE):
        gH_lb = gl_b[:, l].conj().T  # Conjugate transpose of BS to ER node l channel gain
        gH_lr_Phi = np.dot(gl_r[:, l].conj().T, Phi)  # Conjugate transpose of RIS to ER node l channel gain with Phi
        y_E_l = np.dot(gH_lb + np.dot(gH_lr_Phi, H), pk) + ue[l]  # Received signal at l-th ER node
        received_signals_ER.append(y_E_l)

        # Calculate harvested energy
        M_l = np.dot(np.diag(gl_r[:, l].conj()), H)  # Cascaded channel
        harvested_energy_l = kappa * np.sum([np.abs(np.dot(gH_lb + np.dot(theta.conj().T, M_l), pk[:, k])) ** 2 for k in range(KI)])
        harvested_energy.append(harvested_energy_l)

    return received_signals_ER, harvested_energy

# Define harvesting efficiency
kappa = 0.5

# Compute received signals and harvested energy for all ER nodes
received_signals_ER, harvested_energy = compute_received_signal_and_energy(gl_b, gl_r, H, Phi, pk, ue, kappa)


average_efficiency = np.linspace(0.01, 0.5, 15)  # Example range of average efficiency
harvestedd_energy = np.random.uniform(1, 16, size=15)  # Example range of harvested energy

# Plotting the graph as a line plot
plt.figure(figsize=(10, 6))
plt.plot(average_efficiency, harvestedd_energy, marker='o', color='blue', linestyle='-', linewidth=2, markersize=8, label='Energy Harvested')
plt.xlabel('Average Efficiency')
plt.ylabel('Energy Harvested')
plt.title('Energy Harvested vs. Average Efficiency')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
