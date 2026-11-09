import matplotlib.pyplot as plt
import numpy as np


def plot_blockade_comparison(runs, omega):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for (periods, times, rr_curves, analytical_rr), ax in zip(runs, axes):
        for V_ratio, p_rr in rr_curves.items():
            ax.plot(omega * times, p_rr, label=rf"$V/\Omega={V_ratio}$")
            if V_ratio == 0:
                ax.plot(omega * times, analytical_rr, "k--", linewidth=1.5,
                        label=r"$V=0$ analytical")
        ax.set_title(rf"Evolution to $\Omega t={periods}\pi$")
        ax.set_xlabel(r"$\Omega t$")
        ax.set_ylabel(r"$P_{rr}$")
        ax.legend()
    fig.tight_layout()


def plot_collective_excitation(times, p_gg, p_W, ideal_tau, omega):
    plt.figure(figsize=(7, 4))
    plt.plot(omega * times, p_gg, label=r"$P_{gg}$")
    plt.plot(omega * times, p_W, label=r"$P_W$")
    plt.plot(omega * times, np.sin(np.sqrt(2) * omega * times / 2) ** 2,
             "k--", label=r"Ideal $P_W$ ($\Omega_{\rm eff}=\sqrt{2}\Omega$)")
    plt.axvline(ideal_tau, color="gray", linestyle=":", label=r"$\tau=\pi/\sqrt{2}$")
    plt.xlabel(r"$\tau=\Omega t$")
    plt.ylabel("Population")
    plt.title(r"Collective excitation at $V/\Omega=20$")
    plt.legend()
    plt.tight_layout()


def plot_fidelity_scan(times, curves, strengths, peak_fidelities,
                       fixed_fidelities, best_pulse_taus, ideal_tau, omega):
    fig, (fidelity_ax, peak_ax, pulse_ax) = plt.subplots(1, 3, figsize=(17, 4))
    for V_ratio, fidelity in curves.items():
        fidelity_ax.plot(omega * times, fidelity, label=rf"$V/\Omega={V_ratio}$")
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
    fig.tight_layout()


def plot_decay(times, curves, ideal_tau, omega):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for gamma_ratio, populations, fidelity in curves:
        label = rf"$\Gamma/\Omega={gamma_ratio:g}$"
        axes[0].plot(omega * times, populations[0], label=label)
        axes[1].plot(omega * times, populations[3], label=label)
        axes[2].plot(omega * times, fidelity, label=label)
    for ax, title, ylabel in zip(axes,
                                 [r"Ground state", r"Double excitation", r"W-state preparation"],
                                 [r"$P_{gg}$", r"$P_{rr}$", r"$F_W$"]):
        ax.set_title(title)
        ax.set_xlabel(r"$\tau=\Omega t$")
        ax.set_ylabel(ylabel)
        ax.legend()
    axes[2].axvline(ideal_tau, color="gray", linestyle=":")
    fig.tight_layout()


def plot_noise(times, curves, final_single, final_fidelity, ideal_tau, omega):
    fig, (noise_ax, final_ax) = plt.subplots(1, 2, figsize=(12, 4))
    for label, fidelity in curves:
        noise_ax.plot(omega * times, fidelity, label=label)
    noise_ax.axvline(ideal_tau, color="gray", linestyle=":")
    noise_ax.set_xlabel(r"$\tau=\Omega t$")
    noise_ax.set_ylabel(r"$F_W$")
    noise_ax.legend()
    positions = np.arange(len(curves))
    final_ax.bar(positions - 0.18, final_single, width=0.36, label=r"$P_{gr}+P_{rg}$")
    final_ax.bar(positions + 0.18, final_fidelity, width=0.36, label=r"$F_W$")
    final_ax.set_xticks(positions, [label for label, _ in curves])
    final_ax.set_ylim(0, 1.05)
    final_ax.set_ylabel(r"At fixed $\tau=\pi/\sqrt{2}$")
    final_ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2)
    fig.tight_layout()


def plot_dephasing_heatmap(interaction_values, dephasing_values, fidelity_map):
    fig, ax = plt.subplots(figsize=(8, 4))
    image = ax.pcolormesh(interaction_values, dephasing_values, fidelity_map,
                          shading="nearest", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, label=r"$F_W$")
    ax.set_xlabel(r"$V/\Omega$")
    ax.set_ylabel(r"$\gamma_\phi/\Omega$")
    ax.set_title(r"Fixed pulse $\tau=\pi/\sqrt{2}$, no spontaneous decay")
    fig.tight_layout()


def plot_laser_heatmap(detuning_values, duration_errors, laser_map,
                       best_row, best_column):
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.pcolormesh(detuning_values, duration_errors, laser_map,
                          shading="nearest", cmap="viridis", vmin=0.7, vmax=1)
    fig.colorbar(image, ax=ax, label=r"Final $F_W$")
    ax.plot(0, 0, "wx", markersize=9, markeredgewidth=2, label="Nominal pulse")
    ax.plot(detuning_values[best_column], duration_errors[best_row], "r*",
            markersize=12, label="Best sampled")
    ax.set_xlabel(r"Detuning $\Delta/\Omega$")
    ax.set_ylabel(r"Relative duration error $(t-t_0)/t_0$")
    ax.set_title(r"$V/\Omega=20$, $t_0=\pi/(\sqrt{2}\Omega)$, no decoherence")
    ax.legend()
    fig.tight_layout()
