import matplotlib.pyplot as plt
import numpy as np
from qutip import basis, sesolve, sigmax


g = basis(2, 0)
r = basis(2, 1)

Omega = 1.0
H = Omega * sigmax() / 2
times = np.linspace(0, 4 * np.pi / Omega, 400)

result = sesolve(H, g, times, e_ops=[g * g.dag(), r * r.dag()])

plt.plot(times, result.expect[0], label=r"$P_g$")
plt.plot(times, result.expect[1], label=r"$P_r$")
plt.xlabel("Time")
plt.ylabel("Population")
plt.legend()
plt.tight_layout()
plt.show()
