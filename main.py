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
plt.show()
