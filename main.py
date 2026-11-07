import matplotlib.pyplot as plt
import numpy as np
from qutip import basis, expect, mesolve, qeye, sesolve, sigmax, tensor


g = basis(2, 0)
r = basis(2, 1)

Omega = 1.0
gg = tensor(g, g)
gr = tensor(g, r)
rg = tensor(r, g)
rr = tensor(r, r)
W = (gr + rg) / np.sqrt(2)
W_projector = W * W.dag()
states = [gg, gr, rg, rr]
projectors = [state * state.dag() for state in states]
n = r * r.dag()

H_laser = Omega / 2 * (tensor(sigmax(), qeye(2)) + tensor(qeye(2), sigmax()))
interaction = tensor(n, n)


def make_hamiltonian(V_ratio):
    return H_laser + V_ratio * Omega * interaction


def run_simulation(V_ratio, times):
    return sesolve(make_hamiltonian(V_ratio), gg, times)


def calculate_populations(result):
    return np.real(np.array(expect(projectors, result.states)))


def calculate_w_fidelity(result):
    if result.states[0].isket:
        return np.array([abs(W.overlap(state)) ** 2 for state in result.states])
    # For a density matrix, fidelity with the pure target is Tr(rho |W><W|).
    return np.real(np.array(expect(W_projector, result.states)))


fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

for periods, samples, ax in zip([4, 20], [400, 2000], axes):
    times = np.linspace(0, periods * np.pi / Omega, samples)

    for V_ratio in [0, 1, 5, 20]:
        result = run_simulation(V_ratio, times)
        populations = calculate_populations(result)
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
collective_result = run_simulation(20, collective_times)
p_gg = calculate_populations(collective_result)[0]
p_W = calculate_w_fidelity(collective_result)
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
fidelity_fig, (fidelity_ax, peak_ax, pulse_ax) = plt.subplots(1, 3, figsize=(17, 4))
strengths = [0, 1, 5, 20]
peak_fidelities = []
fixed_fidelities = []
best_pulse_taus = []

for V_ratio in strengths:
    result = run_simulation(V_ratio, fidelity_times)
    fidelity = calculate_w_fidelity(result)
    peak_index = np.argmax(fidelity)
    peak_fidelities.append(fidelity[peak_index])
    best_pulse_taus.append(Omega * fidelity_times[peak_index])
    fixed_result = run_simulation(V_ratio, [0, ideal_tau / Omega])
    fixed_fidelity = calculate_w_fidelity(fixed_result)[-1]
    fixed_fidelities.append(fixed_fidelity)
    print(f"V/Ω = {V_ratio:g}: max F_W = {fidelity[peak_index]:.4f} "
          f"at τ = {best_pulse_taus[-1]:.4f}; "
          f"F_W at τ = π/√2: {fixed_fidelity:.4f}")
    fidelity_ax.plot(Omega * fidelity_times, fidelity, label=rf"$V/\Omega={V_ratio}$")

fidelity_ax.set_xlabel(r"$\tau=\Omega t$")
fidelity_ax.set_ylabel(r"$F_W=|\langle W|\psi\rangle|^2$")
fidelity_ax.legend()
peak_ax.plot(strengths, peak_fidelities, "o-", label="Best sampled pulse")
peak_ax.plot(strengths, fixed_fidelities, "s--", label=r"Fixed $\tau=\pi/\sqrt{2}$")
peak_ax.set_title(r"Pulse scan: $0\leq\tau\leq4\pi$")
peak_ax.set_xlabel(r"$V/\Omega$")
peak_ax.set_ylabel(r"$F_W$")
peak_ax.set_ylim(0, 1.05)
peak_ax.legend()
pulse_ax.plot(strengths, best_pulse_taus, "o-")
pulse_ax.axhline(ideal_tau, color="gray", linestyle="--",
                 label=r"Strong blockade $\pi/\sqrt{2}$")
pulse_ax.set_xlabel(r"$V/\Omega$")
pulse_ax.set_ylabel(r"Best pulse duration $\tau$")
pulse_ax.legend()
fidelity_fig.tight_layout()

decay_times = np.linspace(0, 4 * np.pi / Omega, 1200)
unitary_result = run_simulation(20, decay_times)
unitary_populations = calculate_populations(unitary_result)
unitary_fidelity = calculate_w_fidelity(unitary_result)
decay_fig, decay_axes = plt.subplots(1, 3, figsize=(15, 4))
decay = g * r.dag()

for gamma_ratio in [0, 0.01, 0.05]:
    # Independent Markovian decay on each atom; no dephasing or collective emission.
    c_ops = []
    if gamma_ratio > 0:
        rate = gamma_ratio * Omega
        c_ops = [np.sqrt(rate) * tensor(decay, qeye(2)),
                 np.sqrt(rate) * tensor(qeye(2), decay)]
    result = mesolve(make_hamiltonian(20), gg * gg.dag(), decay_times, c_ops=c_ops)
    populations = calculate_populations(result)
    fidelity = calculate_w_fidelity(result)
    trace_error = np.max(np.abs(populations.sum(axis=0) - 1))
    print(f"Γ/Ω = {gamma_ratio:g}: max F_W = {fidelity.max():.4f}, "
          f"population error = {trace_error:.2e}")
    assert trace_error < 1e-8

    if gamma_ratio == 0:
        population_difference = np.max(np.abs(populations - unitary_populations))
        fidelity_difference = np.max(np.abs(fidelity - unitary_fidelity))
        print(f"No-decay mesolve vs sesolve: population difference = "
              f"{population_difference:.2e}, fidelity difference = {fidelity_difference:.2e}")
        assert population_difference < 1e-4
        assert fidelity_difference < 1e-4

    label = rf"$\Gamma/\Omega={gamma_ratio:g}$"
    decay_axes[0].plot(Omega * decay_times, populations[0], label=label)
    decay_axes[1].plot(Omega * decay_times, populations[3], label=label)
    decay_axes[2].plot(Omega * decay_times, fidelity, label=label)

for ax, title, ylabel in zip(decay_axes,
                             [r"Ground state", r"Double excitation", r"W-state preparation"],
                             [r"$P_{gg}$", r"$P_{rr}$", r"$F_W$"]):
    ax.set_title(title)
    ax.set_xlabel(r"$\tau=\Omega t$")
    ax.set_ylabel(ylabel)
    ax.legend()
decay_axes[2].axvline(ideal_tau, color="gray", linestyle=":")
decay_fig.tight_layout()
plt.show()
