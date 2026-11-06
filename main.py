import matplotlib.pyplot as plt
import numpy as np
from qutip import basis, qeye, sesolve, sigmax, tensor


g = basis(2, 0)
r = basis(2, 1)

Omega = 1.0
gg = tensor(g, g)
gr = tensor(g, r)
rg = tensor(r, g)
rr = tensor(r, r)
W = (gr + rg) / np.sqrt(2)
states = [gg, gr, rg, rr]
projectors = [state * state.dag() for state in states]
n = r * r.dag()

H_laser = Omega / 2 * (tensor(sigmax(), qeye(2)) + tensor(qeye(2), sigmax()))
interaction = tensor(n, n)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

for periods, samples, ax in zip([4, 20], [400, 2000], axes):
    times = np.linspace(0, periods * np.pi / Omega, samples)

    for V_ratio in [0, 1, 5, 20]:
        H = H_laser + V_ratio * Omega * interaction
        result = sesolve(H, gg, times, e_ops=projectors)
        populations = np.array(result.expect)
        max_error = np.max(np.abs(populations.sum(axis=0) - 1))
        print(f"End time = {periods}π/Ω, V/Ω = {V_ratio:g}: "
              f"max P_rr = {populations[3].max():.3f}, "
              f"population error = {max_error:.2e}")
        assert max_error < 1e-8

        ax.plot(Omega * times, populations[3], label=rf"$V/\Omega={V_ratio}$")
        if V_ratio == 0:
            analytical_rr = np.sin(Omega * times / 2) ** 4
            max_difference = np.max(np.abs(populations[3] - analytical_rr))
            print(f"End time = {periods}π/Ω: max numerical-analytical "
                  f"difference = {max_difference:.2e}")
            assert max_difference < 1e-4
            ax.plot(Omega * times, analytical_rr, "k--", linewidth=1.5,
                    label=r"$V=0$ analytical")

    ax.set_title(rf"Evolution to $\Omega t={periods}\pi$")
    ax.set_xlabel(r"$\Omega t$")
    ax.set_ylabel(r"$P_{rr}$")
    ax.legend()

plt.tight_layout()

ideal_tau = np.pi / np.sqrt(2)
collective_times = np.linspace(0, 2 * ideal_tau / Omega, 1200)
H_strong = H_laser + 20 * Omega * interaction
collective_result = sesolve(H_strong, gg, collective_times,
                            e_ops=[gg * gg.dag(), W * W.dag()])
p_gg, p_W = collective_result.expect
first_peak = np.argmax(p_W)
peak_tau = Omega * collective_times[first_peak]
print(f"First W maximum: τ = {peak_tau:.4f}, P_W = {p_W[first_peak]:.4f}")
print(f"Ideal τ = π/√2 = {ideal_tau:.4f}; difference = {abs(peak_tau - ideal_tau):.4f}")
print(f"Inferred Ω_eff/Ω = {np.pi / peak_tau:.4f}; expected √2 = {np.sqrt(2):.4f}")
assert abs(peak_tau - ideal_tau) < 0.1

plt.figure(figsize=(7, 4))
plt.plot(Omega * collective_times, p_gg, label=r"$P_{gg}$")
plt.plot(Omega * collective_times, p_W, label=r"$P_W$")
plt.plot(Omega * collective_times,
         np.sin(np.sqrt(2) * Omega * collective_times / 2) ** 2,
         "k--", label=r"Ideal $P_W$ ($\Omega_{\rm eff}=\sqrt{2}\Omega$)")
plt.axvline(ideal_tau, color="gray", linestyle=":", label=r"$\tau=\pi/\sqrt{2}$")
plt.xlabel(r"$\tau=\Omega t$")
plt.ylabel("Population")
plt.title(r"Collective excitation at $V/\Omega=20$")
plt.legend()
plt.tight_layout()

fidelity_times = np.linspace(0, 4 * np.pi / Omega, 1200)
fidelity_fig, (fidelity_ax, peak_ax) = plt.subplots(1, 2, figsize=(12, 4))
strengths = [0, 1, 5, 20]
peak_fidelities = []

for V_ratio in strengths:
    H = H_laser + V_ratio * Omega * interaction
    result = sesolve(H, gg, fidelity_times)
    fidelity = np.array([abs(W.overlap(state)) ** 2 for state in result.states])
    peak_index = np.argmax(fidelity)
    peak_fidelities.append(fidelity[peak_index])
    print(f"V/Ω = {V_ratio:g}: max F_W = {fidelity[peak_index]:.4f} "
          f"at τ = {Omega * fidelity_times[peak_index]:.4f}")
    fidelity_ax.plot(Omega * fidelity_times, fidelity, label=rf"$V/\Omega={V_ratio}$")

fidelity_ax.set_xlabel(r"$\tau=\Omega t$")
fidelity_ax.set_ylabel(r"$F_W=|\langle W|\psi\rangle|^2$")
fidelity_ax.legend()
peak_ax.plot(strengths, peak_fidelities, "o-")
peak_ax.set_xlabel(r"$V/\Omega$")
peak_ax.set_ylabel(r"Maximum $F_W$ for $0\leq\tau\leq4\pi$")
peak_ax.set_ylim(0, 1.05)
fidelity_fig.tight_layout()
plt.show()
