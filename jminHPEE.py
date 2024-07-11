import numpy as np
import matplotlib.pyplot as plt

# Parameters (assuming values for demonstration)
MB = int(input("Enter MB " )) # Number of antennas at BS
KI = int(input("Enter KI " )) # Number of IR nodes
KE = 4  # Number of ER nodes
N = 20  # Number of reflecting elements in RIS
Pmax = 10  # Maximum transmit power available at BS in Watts
Rmin = 0.2  # Minimum rate for IR nodes in bits/s/Hz
Jmin_l_range = np.linspace(5, 15, 20)  # Range of Jmin,l values for plotting

# Function to compute harvested energy given Jmin,l
def compute_harvested_energy(Jmin_l, kappa):
    # Example calculation, replace with actual computation if needed
    return np.random.uniform(Jmin_l, Jmin_l + 0.5, size=KE)

# Function to compute energy efficiency given harvested energy and total power consumed
def compute_energy_efficiency(harvested_energy, total_power_consumed):
    # Example calculation, replace with actual computation if needed
    return np.random.uniform(0.5, 1.5)

# Dummy calculations (replace with actual values from your computations)
harvested_energy_values = [compute_harvested_energy(Jmin_l, 0.5) for Jmin_l in Jmin_l_range]
energy_efficiency_values = [compute_energy_efficiency(he, 100) for he in harvested_energy_values]

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plotting harvested power on ax1 (left y-axis)
ax1.plot(Jmin_l_range, harvested_energy_values, marker='o', linestyle='-', color='blue', label='Harvested Power (HP)')
ax1.set_xlabel('Jmin,l')
ax1.set_ylabel('Harvested Power (HP)', color='red')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True)

# Creating a second y-axis for energy efficiency
ax2 = ax1.twinx()
ax2.plot(Jmin_l_range, energy_efficiency_values, marker='s', linestyle='--', color='green', label='Energy Efficiency (EE)')
ax2.set_ylabel('Energy Efficiency (EE)', color='green')
ax2.tick_params(axis='y', labelcolor='green')

# Title and legend
plt.title('Average Energy Efficiency and Harvested Power vs. Jmin,l')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Adjust layout
plt.tight_layout()
plt.show()
