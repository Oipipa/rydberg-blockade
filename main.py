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
decay = g * r.dag()

H_laser = Omega / 2 * (tensor(sigmax(), qeye(2)) + tensor(qeye(2), sigmax()))
excitation = tensor(n, qeye(2)) + tensor(qeye(2), n)
interaction = tensor(n, n)


def make_hamiltonian(V_ratio, delta_ratio=0):
    return H_laser - delta_ratio * Omega * excitation + V_ratio * Omega * interaction


def run_simulation(V_ratio, times, delta_ratio=0):
    return sesolve(make_hamiltonian(V_ratio, delta_ratio), gg, times)


def run_open_simulation(V_ratio, times, gamma_ratio=0, dephasing_ratio=0, delta_ratio=0):
    # Independent Markovian channels on the two atoms; no collective emission.
    c_ops = []
    if gamma_ratio > 0:
        rate = gamma_ratio * Omega
        c_ops += [np.sqrt(rate) * tensor(decay, qeye(2)),
                  np.sqrt(rate) * tensor(qeye(2), decay)]
    if dephasing_ratio > 0:
        rate = dephasing_ratio * Omega
        c_ops += [np.sqrt(2 * rate) * tensor(n, qeye(2)),
                  np.sqrt(2 * rate) * tensor(qeye(2), n)]
    return mesolve(make_hamiltonian(V_ratio, delta_ratio), gg * gg.dag(), times,
                   c_ops=c_ops)


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

for gamma_ratio in [0, 0.01, 0.05]:
    result = run_open_simulation(20, decay_times, gamma_ratio=gamma_ratio)
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

noise_times = np.linspace(0, 2 * ideal_tau / Omega, 1201)
final_index = len(noise_times) // 2
noise_fig, (noise_ax, final_ax) = plt.subplots(1, 2, figsize=(12, 4))
noise_cases = [("No noise", 0, 0), ("Decay", 0.05, 0),
               ("Dephasing", 0, 0.05), ("Both", 0.05, 0.05)]
final_single = []
final_fidelity = []

for label, gamma_ratio, dephasing_ratio in noise_cases:
    result = run_open_simulation(20, noise_times, gamma_ratio, dephasing_ratio)
    populations = calculate_populations(result)
    fidelity = calculate_w_fidelity(result)
    # F_W = (P_gr + P_rg)/2 + Re(rho_gr,rg), so population alone misses lost coherence.
    single_excitation = populations[1] + populations[2]
    assert np.max(np.abs(populations.sum(axis=0) - 1)) < 1e-8
    final_single.append(single_excitation[final_index])
    final_fidelity.append(fidelity[final_index])
    print(f"{label} (Γ/Ω={gamma_ratio:g}, γφ/Ω={dephasing_ratio:g}): "
          f"P_single={final_single[-1]:.4f}, F_W={final_fidelity[-1]:.4f} "
          f"at τ=π/√2")
    noise_ax.plot(Omega * noise_times, fidelity, label=label)

noise_ax.axvline(ideal_tau, color="gray", linestyle=":")
noise_ax.set_xlabel(r"$\tau=\Omega t$")
noise_ax.set_ylabel(r"$F_W$")
noise_ax.legend()
positions = np.arange(len(noise_cases))
final_ax.bar(positions - 0.18, final_single, width=0.36, label=r"$P_{gr}+P_{rg}$")
final_ax.bar(positions + 0.18, final_fidelity, width=0.36, label=r"$F_W$")
final_ax.set_xticks(positions, [case[0] for case in noise_cases])
final_ax.set_ylim(0, 1.05)
final_ax.set_ylabel(r"At fixed $\tau=\pi/\sqrt{2}$")
final_ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2)
noise_fig.tight_layout()

interaction_values = np.linspace(0, 20, 21)
dephasing_values = np.linspace(0, 0.1, 11)
fidelity_map = np.empty((len(dephasing_values), len(interaction_values)))

for row, dephasing_ratio in enumerate(dephasing_values):
    for column, V_ratio in enumerate(interaction_values):
        result = run_open_simulation(V_ratio, [0, ideal_tau / Omega],
                                     dephasing_ratio=dephasing_ratio)
        fidelity_map[row, column] = calculate_w_fidelity(result)[-1]

for V_ratio, fixed_fidelity in zip(strengths, fixed_fidelities):
    assert abs(fidelity_map[0, int(V_ratio)] - fixed_fidelity) < 1e-4
print(f"At V/Ω=20, F_W changes from {fidelity_map[0, -1]:.4f} "
      f"to {fidelity_map[-1, -1]:.4f} as γφ/Ω rises to 0.1")

heatmap_fig, heatmap_ax = plt.subplots(figsize=(8, 4))
image = heatmap_ax.pcolormesh(interaction_values, dephasing_values, fidelity_map,
                             shading="nearest", vmin=0, vmax=1)
heatmap_fig.colorbar(image, ax=heatmap_ax, label=r"$F_W$")
heatmap_ax.set_xlabel(r"$V/\Omega$")
heatmap_ax.set_ylabel(r"$\gamma_\phi/\Omega$")
heatmap_ax.set_title(r"Fixed pulse $\tau=\pi/\sqrt{2}$, no spontaneous decay")
heatmap_fig.tight_layout()

detuning_values = np.linspace(-0.5, 0.5, 41)
duration_errors = np.linspace(-0.2, 0.2, 41)
laser_map = np.empty((len(duration_errors), len(detuning_values)))

for row, duration_error in enumerate(duration_errors):
    pulse_time = ideal_tau * (1 + duration_error) / Omega
    for column, delta_ratio in enumerate(detuning_values):
        result = run_simulation(20, [0, pulse_time], delta_ratio=delta_ratio)
        laser_map[row, column] = calculate_w_fidelity(result)[-1]

center_row = len(duration_errors) // 2
center_column = len(detuning_values) // 2
center_fidelity = laser_map[center_row, center_column]
assert abs(center_fidelity - fixed_fidelities[-1]) < 1e-4
best_row, best_column = np.unravel_index(np.argmax(laser_map), laser_map.shape)
worst_row, worst_column = np.unravel_index(np.argmin(laser_map), laser_map.shape)
print(f"Laser error scan: F_W={center_fidelity:.4f} at zero error; "
      f"best F_W={laser_map[best_row, best_column]:.4f} at "
      f"Δ/Ω={detuning_values[best_column]:+.3f}, "
      f"duration error={duration_errors[best_row]:+.3f}")
print(f"Lowest sampled F_W={laser_map[worst_row, worst_column]:.4f} at "
      f"Δ/Ω={detuning_values[worst_column]:+.3f}, "
      f"duration error={duration_errors[worst_row]:+.3f}")
print(f"At zero detuning, F_W at -20%/+20% duration error: "
      f"{laser_map[0, center_column]:.4f}/{laser_map[-1, center_column]:.4f}")
print(f"At zero duration error, F_W at -0.5/+0.5 detuning: "
      f"{laser_map[center_row, 0]:.4f}/{laser_map[center_row, -1]:.4f}")

laser_fig, laser_ax = plt.subplots(figsize=(8, 5))
image = laser_ax.pcolormesh(detuning_values, duration_errors, laser_map,
                           shading="nearest", cmap="viridis", vmin=0.7, vmax=1)
laser_fig.colorbar(image, ax=laser_ax, label=r"Final $F_W$")
laser_ax.plot(0, 0, "wx", markersize=9, markeredgewidth=2, label="Nominal pulse")
laser_ax.plot(detuning_values[best_column], duration_errors[best_row], "r*",
              markersize=12, label="Best sampled")
laser_ax.set_xlabel(r"Detuning $\Delta/\Omega$")
laser_ax.set_ylabel(r"Relative duration error $(t-t_0)/t_0$")
laser_ax.set_title(r"$V/\Omega=20$, $t_0=\pi/(\sqrt{2}\Omega)$, no decoherence")
laser_ax.legend()
laser_fig.tight_layout()
plt.show()
