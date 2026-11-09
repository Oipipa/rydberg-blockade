import numpy as np
from qutip import basis, expect, mesolve, qeye, sesolve, sigmax, tensor


Omega = 1.0
g = basis(2, 0)
r = basis(2, 1)
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
