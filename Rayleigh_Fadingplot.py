import numpy as np
import matplotlib.pyplot as plt

# Function to convert spherical coordinates to Cartesian coordinates


def spherical_to_cartesian(d, theta, phi):
    """
    Convert spherical coordinates to Cartesian coordinates.
    - d : Distance
    - theta : Zenith angle in radians
    - phi: Azimuth angle in radians
    """
    Cx = d * np.sin(theta) * np.cos(phi)
    Cy = d * np.sin(theta) * np.sin(phi)
    Cz = d * np.cos(theta)
    return Cx, Cy, Cz

# Function to calculate distance between two points in Cartesian coordinates


def calculate_distance(point1, point2):
    """
    distance between two points in Cartesian coordinates.
    - point1: (x1, y1, z1)
    - point2: (x2, y2, z2)
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

# Constants
Pt = 1.0  # Power Transmitted in Watts
Gt = 1.0  # Gain of Transmitter antenna
Gr = 1.0  # Gain of Receiver antenna
Ge_mn = 0.9  # Gain of the element Em,n
Gamma_mn = 0.5  # Reflection coefficient of the element Em,n
eff = 0.8  # Efficiency
N_p = 100 #No of packets

# RIS element parameters
M = 4  # Number of rows
N = 3  # Number of columns
dx = 0.5  # Width of element in meters
dy = 0.5  # Height of element in meters

# Element effective areas
Aet_mn = 0.1
Aer_mn = 0.1
Ar = 0.1

# Example coordinates for Tx and Rx
d1 = 10.0  # Distance of transmitter from RIS
theta1 = np.pi / 4  # Zenith angle of transmitter
phi1 = np.pi / 2  # Azimuth angle of transmitter
d2 = 8.0  # Distance of receiver from RIS
theta2 = np.pi / 3  # Zenith angle of receiver
phi2 = np.pi / 4  # Azimuth angle of receiver

# Calculate Cartesian coordinates of transmitter and receiver
Tx_Cartesian = spherical_to_cartesian(d1, theta1, phi1)
Rx_Cartesian = spherical_to_cartesian(d2, theta2, phi2)

# Assuming a specific element Em,n
m = 2  # Row index of the element
n = 1  # Column index of the element

# Calculate Cartesian coordinates of the selected RIS element Em,n
pm_n = ((n - (N + 1) / 2) * dx, (M + 1) / 2 - m * dy, 0.0)

# Calculate distances between Tx, Em,n and Rx, Em,n
dt_mn = calculate_distance(Tx_Cartesian, pm_n)
dr_mn = calculate_distance(Rx_Cartesian, pm_n)

# Power density generated by the transmitter on the element Em,n
St_mn = (Pt * Gt) / (4 * np.pi * dt_mn**2)

# Calculate power captured by Em,n
Pt_mn = St_mn * Aet_mn

# Calculate power density generated by Em,n
Sr_mn = Pt_mn * Ge_mn / (4 * np.pi * dr_mn**2)

# Calculate power received from Em,n
Pr_mn = Sr_mn * Ar

# Final received power reflected from m,nth element
P_r = (Pt * Gt * Gr * Aer_mn * Aet_mn) / (((4 * np.pi * dt_mn * dr_mn)**2) * eff)

# Friss equation calculation
d = calculate_distance(Tx_Cartesian, Rx_Cartesian)
c = 3e8
f = 2.4e9
lambda_ = c / f
p_rfriss = Pt * Gt * Gr * (lambda_ / (4 * np.pi * d)) ** 2

# Print results
print("*********************************************************************")
print(" ")
print("Distance from Tx to Em,n (dt_mn):", dt_mn, "m")
print("Distance from Rx to Em,n (dr_mn):", dr_mn, "m")
print("------------------------------------------------------------")
print("Transmitted Power:", Pt, "W")
print("Final Received power reflected from m,nth element:", P_r, "W")
print("Final Received Average power for N packets :", P_r/N_p, "W")

print("------------------------------------------------------------")
print("Distance from Tx to Rx:", d, "m")
print("Result from Friss equation (if clear LOS):", p_rfriss, "W")
print(" ")
print("*********************************************************************")

# Plot distance vs. performance criteria
distances = np.linspace(1, 50, 500)  # Distances from 1 to 50 meters
powers_friss = []
powers_ris = []
path_loss_friss = []
path_loss_ris = []

# Rayleigh fading parameters
sigma = 1.0  # Standard deviation for Rayleigh fading

for distance in distances:
    # Cartesian coordinates for this distance
    Rx_Cartesian_dynamic = spherical_to_cartesian(distance, theta2, phi2)

    # Dynamic distance from Tx to Rx
    d_dynamic = calculate_distance(Tx_Cartesian, Rx_Cartesian_dynamic)

    # Rayleigh fading gain (assuming complex fading)
    fading_gain = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)

    # Power received using Friss equation with Rayleigh fading
    # np.abs(fading_gain) ** 2: Computes the power gain due to Rayleigh fading
    p_r_friss_dynamic = (Pt * Gt * Gr * (lambda_ / (4 * np.pi * d_dynamic))) ** 2 * (np.abs(fading_gain) ** 2)
    powers_friss.append(p_r_friss_dynamic)

    # Path loss using Friss equation with Rayleigh fading
    path_loss_friss_dynamic = 10 * np.log10(Pt / p_r_friss_dynamic)
    path_loss_friss.append(path_loss_friss_dynamic)

    # Dynamic distance to RIS element
    dr_mn_dynamic = calculate_distance(Rx_Cartesian_dynamic, pm_n)

    # Final received power reflected from m,nth element with Rayleigh fading
    P_r_dynamic =( (Pt * Gt * Gr) / (((4 * np.pi * dt_mn * dr_mn_dynamic) ** 2) * eff) )* (np.abs(fading_gain) ** 2)
    powers_ris.append(P_r_dynamic)

    # Path loss using RIS reflection with Rayleigh fading
    path_loss_ris_dynamic = 10 * np.log10(Pt / P_r_dynamic)
    path_loss_ris.append(path_loss_ris_dynamic)

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plot power received
axs[0].plot(distances, powers_friss, label='Friss Equation + Rayleigh Fading', marker='o', linestyle='-', color='brown')
axs[0].plot(distances, powers_ris, label='RIS Reflected Power + Rayleigh Fading', marker='o', linestyle='-', color='yellow')
axs[0].set_xlabel('Distance (m)', fontsize=12)
axs[0].set_ylabel('Power Received (W)', fontsize=12)
axs[0].set_title('Power Received (Friss vs. RIS with Rayleigh Fading)', fontsize=14)
axs[0].legend(fontsize=10)
axs[0].grid(True, alpha=0.5)


# Plot path loss
axs[1].plot(distances, path_loss_friss, label='Friss Equation + Rayleigh Fading', marker='o', linestyle='-', color='orange' )
axs[1].plot(distances, path_loss_ris, label='RIS Reflected Power + Rayleigh Fading', marker='o', linestyle='-', color='blue')
axs[1].set_xlabel('Distance (m)', fontsize=12)
axs[1].set_ylabel('Path Loss (dB)', fontsize=12)
axs[1].set_title('Path Loss (Friss vs. RIS with Rayleigh Fading)', fontsize=14)
axs[1].legend(fontsize=10)
axs[1].grid(True, alpha=0.5)

# Adjust layout
plt.tight_layout()
plt.show()
