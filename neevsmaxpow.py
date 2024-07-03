import numpy as np
import matplotlib.pyplot as plt

# Define your functions here

# Parameters
MB = 2  # Number of antennas at BS
KI = 2  # Number of IR nodes
KE = 4  # Number of ER nodes
N = 50  # Number of reflecting elements in RIS
Rmin = 0.2  # Minimum rate for IR nodes in bits/s/Hz
Jmin = 20e-3  # Minimum harvested energy for ER nodes in Watts
rician_factor = 1  # Example Rician factor
sigma_epsilon = 1e-3
P_owS_B = 10  # dBm
P_owS_I = 10  # dBm

N_simulations = 15  # Number of simulations per configuration
Pmax_values = np.arange(1, 16)  # Range of Pmax from 1 to 15 Watts
avg_efficiencies = []
cumulative_efficiencies = []

for Pmax in Pmax_values:
    efficiencies = []
    for _ in range(N_simulations):
        efficiency = np.random.rand()
        efficiencies.append(efficiency)
    
    avg_efficiency = np.mean(efficiencies)
    avg_efficiencies.append(avg_efficiency)

    cumulative_efficiency = np.cumsum(efficiencies)
    cumulative_efficiencies.append(cumulative_efficiency)

# Plotting
plt.figure(figsize=(10, 6))

# Plot average energy efficiency
plt.plot(Pmax_values, avg_efficiencies, marker='o', linestyle='-', color='b', label='Average Efficiency')

# Plot cumulative efficiency for each simulation
for i in range(N_simulations):
    plt.plot(Pmax_values, cumulative_efficiencies[i], linestyle='--', alpha=0.5, label=f'Cumulative Efficiency - Simulation {i+1}')

plt.xlabel('Maximum Transmit Power (Pmax) in Watts')
plt.ylabel('Energy Efficiency')
plt.title('Average and Cumulative Energy Efficiency vs Maximum Transmit Power')
plt.legend()
plt.grid(True)
plt.show()
