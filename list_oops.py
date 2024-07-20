import numpy as np

class WirelessSystemSimulation:
    def __init__(self, MB, KI, KE, N, Pmax, Rmin, Jmin):
        self.MB = MB
        self.KI = KI
        self.KE = KE
        self.N = N
        self.Pmax = Pmax
        self.Rmin = Rmin
        self.Jmin = Jmin

        self.position_BS = np.array([0, 0])
        self.position_RIS = np.array([5, 2])
        self.position_IRs = np.array([[30, 0] for _ in range(self.KI)])
        self.position_ERs = np.array([[5, 0] for _ in range(self.KE)])

        self.hk_b = self._rayleigh_channel((self.MB, self.KI))
        self.gl_b = self._rayleigh_channel((self.MB, self.KE))
        self.hk_r = self._rician_channel((self.N, self.KI), rician_factor=1)
        self.gl_r = self._rician_channel((self.N, self.KE), rician_factor=1)
        self.H = self._rician_channel((self.N, self.MB), rician_factor=1)

        self.Phi = np.diag(np.exp(1j * 2 * np.pi * np.random.rand(self.N)))
        self.pk = np.random.randn(self.MB, self.KI) + 1j * np.random.randn(self.MB, self.KI)

        self.sigma_epsilon = 1e-3
        self.kappa = 0.5

    def _rician_channel(self, size, rician_factor):
        K = rician_factor
        los_component = np.random.randn(*size) + 1j * np.random.randn(*size)
        nlos_component = np.random.randn(*size) + 1j * np.random.randn(*size)
        return np.sqrt(K / (K + 1)) * los_component + np.sqrt(1 / (K + 1)) * nlos_component

    def _rayleigh_channel(self, size):
        return (np.random.randn(*size) + 1j * np.random.randn(*size)) * np.sqrt(1 / 2)

    def compute_received_signal_IR(self, vk, ue):
        received_signals = []

        for k in range(self.KI):
            gH_lb = self.hk_b[:, k].conj().T
            gH_lr_Phi = np.dot(self.hk_r[:, k].conj().T, self.Phi.conj().T)
            y_I_k = np.dot(gH_lb, np.dot(self.pk, vk)) + np.dot(gH_lr_Phi, np.dot(self.H, np.dot(self.pk, vk))) + ue[k]
            received_signals.append(y_I_k)

        return received_signals

    def compute_received_signal_ER(self, theta, ue):
        received_signals = []
        harvested_energy = []

        for l in range(self.KE):
            gH_lb = self.gl_b[:, l].conj().T
            gH_lr_Phi = np.dot(self.gl_r[:, l].conj().T, self.Phi)
            y_E_l = np.dot(gH_lb + np.dot(gH_lr_Phi, self.H), self.pk) + ue[l]
            received_signals.append(y_E_l)

            M_l = np.dot(np.diag(self.gl_r[:, l].conj()), self.H)
            harvested_energy_l = self.kappa * np.sum(
                [np.abs(np.dot(gH_lb + np.dot(theta.conj().T, M_l), self.pk[:, k])) ** 2 for k in range(self.KI)])
            harvested_energy.append(harvested_energy_l)

        return received_signals, harvested_energy

    def compute_sinr_and_rate(self):
        sinr_values = []
        rate_values = []

        for k in range(self.KI):
            gk_b = self.hk_b[:, k]
            gk_r = np.dot(self.hk_r[:, k].conj(), self.Phi)
            gk = gk_b + np.dot(gk_r, self.H)
            pk_k = self.pk[:, k][:, np.newaxis]

            numerator = np.abs(np.dot(gk.conj(), pk_k)) ** 2

            interference_sum = 0
            for i in range(self.KI):
                if i != k:
                    gi_b = self.hk_b[:, i]
                    gi_r = np.dot(self.hk_r[:, i].conj(), self.Phi)
                    gi = gi_b + np.dot(gi_r, self.H)
                    pk_i = self.pk[:, i][:, np.newaxis]
                    interference_sum += np.abs(np.dot(gi.conj(), pk_i)) ** 2

            sinr_k = numerator / (interference_sum + self.sigma_epsilon)
            sinr_values.append(sinr_k.item())

            rate_k = np.log2(1 + sinr_k.item())
            rate_values.append(rate_k)

        sum_rate = np.sum(rate_values)

        return sinr_values, rate_values, sum_rate

    def compute_total_power_consumed(self, P_owS_B, P_owS_I):
        total_power = np.sum([np.trace(np.outer(self.pk[:, k], self.pk[:, k].conj())) for k in range(self.KI)])
        total_power += P_owS_B + P_owS_I
        return total_power

    def compute_energy_efficiency(self, total_power_consumed, sum_rate):
        return sum_rate / total_power_consumed

# Example usage:
MB = 2
KI = 2
KE = 4
N = 20
Pmax = 10
Rmin = 0.2
Jmin = 20e-3

wireless_sim = WirelessSystemSimulation(MB, KI, KE, N, Pmax, Rmin, Jmin)

vk = np.random.randn(KI, 1) + 1j * np.random.randn(KI, 1)
ue = np.random.randn(KE) + 1j * np.random.randn(KE)

received_signals_IR = wireless_sim.compute_received_signal_IR(vk, ue)
print("Received signals for IR nodes:", received_signals_IR)

theta = np.exp(1j * 2 * np.pi * np.random.rand(N))
received_signals_ER, harvested_energy = wireless_sim.compute_received_signal_ER(theta, ue)
print("Received signals for ER nodes:", received_signals_ER)
print("Harvested energy for ER nodes:", harvested_energy)

sinr_values, rate_values, sum_rate = wireless_sim.compute_sinr_and_rate()
print("SINR values for IR nodes:", sinr_values)
print("Rate values for IR nodes:", rate_values)
print("Sum Rate:", sum_rate)

P_owS_B = 20  # dBm
P_owS_I = 10  # dBm
total_power_consumed = wireless_sim.compute_total_power_consumed(P_owS_B, P_owS_I)
energy_efficiency = wireless_sim.compute_energy_efficiency(total_power_consumed, sum_rate)
print("Total power consumed:", total_power_consumed)
print("Energy efficiency:", energy_efficiency)
