import matplotlib.pyplot as plt
import numpy as np

from plots import (plot_blockade_comparison, plot_collective_excitation,
                   plot_decay, plot_dephasing_heatmap, plot_fidelity_scan,
                   plot_laser_heatmap, plot_noise)
from rydberg import (Omega, calculate_populations, calculate_w_fidelity,
                     run_open_simulation, run_simulation)


blockade_runs = []
for periods, samples in [(4, 400), (20, 2000)]:
    times = np.linspace(0, periods * np.pi / Omega, samples)
    rr_curves = {}
    analytical_rr = None

    for V_ratio in [0, 1, 5, 20]:
        result = run_simulation(V_ratio, times)
        populations = calculate_populations(result)
        max_error = np.max(np.abs(populations.sum(axis=0) - 1))
        print(f"End time = {periods}π/Ω, V/Ω = {V_ratio:g}: "
              f"max P_rr = {populations[3].max():.3f}, "
              f"population error = {max_error:.2e}")
        assert max_error < 1e-8
        rr_curves[V_ratio] = populations[3]

        if V_ratio == 0:
            analytical_rr = np.sin(Omega * times / 2) ** 4
            max_difference = np.max(np.abs(populations[3] - analytical_rr))
            print(f"End time = {periods}π/Ω: max numerical-analytical "
                  f"difference = {max_difference:.2e}")
            assert max_difference < 1e-4

    blockade_runs.append((periods, times, rr_curves, analytical_rr))

plot_blockade_comparison(blockade_runs, Omega)

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
plot_collective_excitation(collective_times, p_gg, p_W, ideal_tau, Omega)

fidelity_times = np.linspace(0, 4 * np.pi / Omega, 1200)
strengths = [0, 1, 5, 20]
fidelity_curves = {}
peak_fidelities = []
fixed_fidelities = []
best_pulse_taus = []

for V_ratio in strengths:
    result = run_simulation(V_ratio, fidelity_times)
    fidelity = calculate_w_fidelity(result)
    fidelity_curves[V_ratio] = fidelity
    peak_index = np.argmax(fidelity)
    peak_fidelities.append(fidelity[peak_index])
    best_pulse_taus.append(Omega * fidelity_times[peak_index])
    fixed_result = run_simulation(V_ratio, [0, ideal_tau / Omega])
    fixed_fidelity = calculate_w_fidelity(fixed_result)[-1]
    fixed_fidelities.append(fixed_fidelity)
    print(f"V/Ω = {V_ratio:g}: max F_W = {fidelity[peak_index]:.4f} "
          f"at τ = {best_pulse_taus[-1]:.4f}; "
          f"F_W at τ = π/√2: {fixed_fidelity:.4f}")

plot_fidelity_scan(fidelity_times, fidelity_curves, strengths, peak_fidelities,
                   fixed_fidelities, best_pulse_taus, ideal_tau, Omega)

decay_times = np.linspace(0, 4 * np.pi / Omega, 1200)
unitary_result = run_simulation(20, decay_times)
unitary_populations = calculate_populations(unitary_result)
unitary_fidelity = calculate_w_fidelity(unitary_result)
decay_curves = []

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

    decay_curves.append((gamma_ratio, populations, fidelity))

plot_decay(decay_times, decay_curves, ideal_tau, Omega)

noise_times = np.linspace(0, 2 * ideal_tau / Omega, 1201)
final_index = len(noise_times) // 2
noise_cases = [("No noise", 0, 0), ("Decay", 0.05, 0),
               ("Dephasing", 0, 0.05), ("Both", 0.05, 0.05)]
noise_curves = []
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
    noise_curves.append((label, fidelity))

plot_noise(noise_times, noise_curves, final_single, final_fidelity, ideal_tau, Omega)

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
plot_dephasing_heatmap(interaction_values, dephasing_values, fidelity_map)

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
plot_laser_heatmap(detuning_values, duration_errors, laser_map,
                   best_row, best_column)

plt.show()
