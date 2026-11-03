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

H = Omega / 2 * (tensor(sigmax(), qeye(2)) + tensor(qeye(2), sigmax()))
times = np.linspace(0, 4 * np.pi / Omega, 400)

result = sesolve(H, gg, times, e_ops=[state * state.dag() for state in states])
populations = np.array(result.expect)
max_error = np.max(np.abs(populations.sum(axis=0) - 1))
print(f"Maximum total-population error: {max_error:.2e}")
assert max_error < 1e-8

for population, label in zip(populations, [r"$P_{gg}$", r"$P_{gr}$", r"$P_{rg}$", r"$P_{rr}$"]):
    plt.plot(times, population, label=label)
plt.xlabel("Time")
plt.ylabel("Population")
plt.legend()
plt.tight_layout()
plt.show()
