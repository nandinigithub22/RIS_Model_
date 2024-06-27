
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

# Rician fading channel
def rician_channel(size, rician_factor):
    K = rician_factor
    los_component = (np.random.randn(*size) + 1j * np.random.randn(*size)) * np.sqrt(K / (K + 1))
    nlos_component = (np.random.randn(*size) + 1j * np.random.randn(*size)) * np.sqrt(1 / (K + 1))
    return los_component + nlos_component

# Rayleigh fading channel
def rayleigh_channel(size):
    return (np.random.randn(*size) + 1j * np.random.randn(*size)) * np.sqrt(1 / 2)

# The term np.sqrt(1 / 2) is used in the generation of Rayleigh fading channel coefficients
# to normalize the power of the fading channels.

# Channel gains
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
s = pk @ vk  # Combined symbol transmitted to all IR nodes, should be (MB, 1)

# Additive noise at ER nodes
ue = np.random.randn(KE, 1) + 1j * np.random.randn(KE, 1)


def compute_received_signal( hk_b, hk_r, gl_b, gl_r, H, Phi, s, ue ):
    # Initialize an array to store received signals for all IR nodes
    received_signals = []

    # Determine the number of IR nodes (KI) and ER nodes (KE)
    KI = hk_b.shape[1]
    KE = gl_b.shape[1]

    # Compute received signal for each IR node
    for k in range(KI):
        gH_lb = hk_b[:, k].conj().T  # Conjugate transpose of BS to IR node channel gain
        gH_lr_Phi = np.dot(hk_r[:, k].conj().T, Phi.conj().T) # Conjugate transpose of RIS to IR node channel gain with Phi
        y_I_k = np.dot(gH_lb, s) + np.dot(gH_lr_Phi, H @ s) + ue  # Received signal at k-th IR node
        received_signals.append(y_I_k)

    return received_signals

# Compute received signals for all IR nodes


received_signals_IR = compute_received_signal(hk_b, hk_r, gl_b, gl_r, H, Phi, s, ue)

# Print received signals for all IR nodes
for k in range(len(received_signals_IR)):
    print(f"Received signal at IR node {k + 1}:\n", received_signals_IR[k])


# Compute P_k for all k
P_k = []
for k in range(KI):
    P_k.append(np.trace(np.outer(pk[:, k], np.conj(pk[:, k]))))


# Sum of P_k
total_transmit_power = np.sum(P_k)

# Apply power constraint
if total_transmit_power > Pmax:
    scaling_factor = np.sqrt(Pmax / total_transmit_power)
    pk = pk * scaling_factor
    P_k = [P * scaling_factor**2 for P in P_k]

print(f"Transmit power vector pk after applying power constraint:\n{pk}")
print("---------------------------------------------------------------------------------------------------------------")

# Printing other parameters for verification
print("\nBS to IR channel gains (hk_b):\n", hk_b)
print("---------------------------------------------------------------------------------------------------------------")

print("\nRIS to IR channel gains (hk_r):\n", hk_r)
print("---------------------------------------------------------------------------------------------------------------")

print("\nBS to ER channel gains (gl_b):\n", gl_b)
print("---------------------------------------------------------------------------------------------------------------")

print("\nRIS to ER channel gains (gl_r):\n", gl_r)
print("---------------------------------------------------------------------------------------------------------------")

print("\nBS to RIS channel gain (H):\n", H)
print("---------------------------------------------------------------------------------------------------------------")

print("\nPhase shift matrix (Phi):\n", Phi)
print("---------------------------------------------------------------------------------------------------------------")

print("\nTransmit power vector (pk):\n", pk)
print("---------------------------------------------------------------------------------------------------------------")

print("\nCombined information symbols (vk):\n", vk)
print("---------------------------------------------------------------------------------------------------------------")



# Function to compute the received signal at each ER node and the harvested energy
def compute_received_signal_and_energy(gl_b, gl_r, H, Phi, pk, ue, kappa):
    received_signals_ER = []
    harvested_energy = []

# .T: This transposes the result of theta.conj(). Transposition switches the rows and columns of a matrix.
# θ^H represents the Hermitian transpose (conjugate transpose) of θ.
# θ = [θ1, · · · , θN ]^T
    for l in range(KE):
        # Received signal at l-th ER node
        gH_lb = gl_b[:, l].conj().T  # Conjugate transpose of BS to ER node l channel gain
        gH_lr_Phi = np.dot(gl_r[:, l].conj().T, Phi)  # Conjugate transpose of RIS to ER node l channel gain with Phi
        y_E_l = np.dot(gH_lb + np.dot(gH_lr_Phi, H), pk) + ue[l]
        received_signals_ER.append(y_E_l)

        # Harvested energy at l-th ER node
        M_l = np.dot(np.diag(gl_r[:, l].conj()), H)  # Cascaded channel
        harvested_energy_l = kappa * np.sum(
            [np.abs(np.dot(gH_lb + np.dot(theta.conj().T, M_l), pk[:, k])) ** 2 for k in range(KI)])
        harvested_energy.append(harvested_energy_l)

    return received_signals_ER, harvested_energy


# Define harvesting efficiency
kappa = 0.5

# Compute received signals and harvested energy for all ER nodes
received_signals_ER, harvested_energy = compute_received_signal_and_energy(gl_b, gl_r, H, Phi, pk, ue, kappa)


print("---------------------------------------------------------------------------------------------------------------")
print("Received Signal at each ER node and Energy Harvested ")
print("---------------------------------------------------------------------------------------------------------------")

# Print received signals and harvested energy
for l in range(KE):
    print(f"Received signal at ER node {l + 1}:\n", received_signals_ER[l])
    print(f"Harvested energy at ER node {l + 1}: {harvested_energy[l]}")

# Sum harvested power across all ER nodes
total_harvested_power = np.sum(harvested_energy)
print(f"\nTotal harvested power across all ER nodes: {total_harvested_power}")


# Check if harvested energy meets the minimum requirement
if all(energy >= Jmin for energy in harvested_energy):
    print("All ER nodes meet the minimum harvested energy requirement.")
else:
    print("Not all ER nodes meet the minimum harvested energy requirement.")


print("---------------------------------------------------------------------------------------------------------------")


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




pk = np.random.randn(MB, KI) + 1j * np.random.randn(MB, KI)

sigma_epsilon = 1e-3

sinr_values, rate_values, sum_rate = compute_sinr_and_rate(hk_b, hk_r, H, Phi, pk, sigma_epsilon)

for k in range(KI):
    print(f"IR node {k + 1}:")
    print(f"  SINR: {sinr_values[k]}")
    print(f"  Rate: {rate_values[k]} bits/s/Hz")
    print()

print(f"Sum Rate: {sum_rate} bits/s/Hz")

# Check if SINR meets the minimum rate requirement
if all(rate >= Rmin for rate in rate_values):
    print("All IR nodes meet the minimum rate requirement.")
else:
    print("Not all IR nodes meet the minimum rate requirement.")

print("\nSINR values for all IR nodes:", sinr_values)
print("Rate values for all IR nodes:", rate_values)
print("---------------------------------------------------------------------------------------------------------------")


# Define static power used at the BS and RIS
P_owS_B = 20  # dBm
P_owS_I = 10  # dBm


# Function to compute total power consumed
def compute_total_power_consumed(pk, P_owS_B, P_owS_I):
    total_power = np.sum([np.trace(np.outer(pk[:, k], pk[:, k].conj().T)) for k in range(KI)])
    total_power += P_owS_B + P_owS_I
    return total_power



# Compute total power consumed
total_power_consumed = np.abs(compute_total_power_consumed(pk, P_owS_B, P_owS_I))


# Compute energy efficiency
energy_efficiency = np.abs(sum_rate / total_power_consumed)

print("---------------------------------------------------------------------------------------------------------------")

print(f"Total power consumed: {total_power_consumed}")
print(f"Energy efficiency:{energy_efficiency}")

print("****************************************************************************************************************")


# Compute energy efficiency
delta_m_values = np.random.rand(N)  # Example random delta_m values for each iteration
energy_efficiency = np.sum(rate_values) / (total_power_consumed * delta_m_values)

# Plotting efficiency vs iterations and delta_m vs iterations
plt.figure(figsize=(10, 6))
iterations = range(1, N +1) 

# Plot energy efficiency
plt.plot(iterations, energy_efficiency, marker='o', linestyle='-', color='b', label='Energy Efficiency')
plt.xlabel('Iterations (N)')
plt.ylabel('Energy Efficiency')
plt.grid(True)
plt.legend()
plt.show()
