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
times = np.linspace(0, 4 * np.pi / Omega, 400)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
labels = [r"$P_{gg}$", r"$P_{gr}$", r"$P_{rg}$", r"$P_{rr}$"]

for V, ax in zip([0.0, 10.0], axes[:2]):
    H = H_laser + V * tensor(n, n)
    result = sesolve(H, gg, times, e_ops=projectors)
    populations = np.array(result.expect)
    max_error = np.max(np.abs(populations.sum(axis=0) - 1))
    print(f"V={V:g}: maximum total-population error = {max_error:.2e}, "
          f"maximum P_rr = {populations[3].max():.3f}")
    assert max_error < 1e-8

    for population, label in zip(populations, labels):
        ax.plot(times, population, label=label)
    ax.set_title(f"V = {V:g}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Population")
    ax.legend()

    axes[2].plot(times, populations[3], label=f"V = {V:g}")

axes[2].set_title(r"Double excitation $P_{rr}$")
axes[2].set_xlabel("Time")
axes[2].set_ylabel(r"$P_{rr}$")
axes[2].legend()
plt.tight_layout()
plt.show()
