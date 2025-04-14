import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
import copy

from make_segments import make_segments
from solve_for_flow import solve_for_flow
from realign_polarity import realign_polarity
from cell_migration import cell_migration
from plot_network import plot_network

np.random.seed(123456789)

branch_rule = 4
branch_alpha = 1.0
Nt = 40
Pin = 4 * 98
Pout = 1 * 98
mu = 3.5e-3
Nn = 40
Nseg = 40
num_cell = 10
cell_size = 5e-6
w2 = 1
w3 = 0.00
w4 = 0.00
w1 = 1 - w2 - w3 - w4

P = np.zeros(Nn)
Q = np.zeros(Nseg)
G = np.zeros(Nseg)
H = np.zeros(Nseg)
tau = np.zeros(Nseg)
L = 10e-6 * np.ones(Nseg)
Ncell = num_cell * np.ones(Nseg, dtype=int)

segments = make_segments(L)

seg_cells = []
for seg in range(Nseg):
    num_cells = int(Ncell[seg])
    polarity_vectors = np.random.randn(2, num_cells)
    norms = np.linalg.norm(polarity_vectors, axis=0)
    polarity_vectors[:, norms == 0] = np.array([1, 0])[:, None]
    polarity_vectors /= np.linalg.norm(polarity_vectors, axis=0)
    migration_indicators = np.zeros(num_cells, dtype=int)
    seg_cells.append({
        'num_cells': num_cells,
        'polarity_vectors': polarity_vectors,
        'migration_indicators': migration_indicators
    })

D = np.zeros(Nseg)
for seg in range(Nseg):
    if Ncell[seg] >= 1:
        D[seg] = Ncell[seg] * cell_size / np.pi
    else:
        D[seg] = 0

G = (np.pi * D**4) / (128 * mu * L)
H = np.zeros_like(D)
mask = D > 0
H[mask] = (32 * mu) / (np.pi * D[mask]**3)
P, Q, tau = solve_for_flow(G, Pin, Pout, H)

new_seg_cells = copy.deepcopy(seg_cells)
for seg in range(Nseg):
    seg_cells, new_seg_cells = realign_polarity(seg, Q, seg_cells, new_seg_cells, w1, w2, w3, w4)
seg_cells = new_seg_cells

plot_network(segments, D, P, Q, seg_cells, tau)

for t in range(1, Nt + 1):
    print(f"Time step {t}/{Nt}")
    migrate = np.zeros(Nseg, dtype=int)
    new_seg_cells = copy.deepcopy(seg_cells)

    for seg in range(Nseg):
        seg_cells, new_seg_cells = realign_polarity(seg, Q, seg_cells, new_seg_cells, w1, w2, w3, w4)
        seg_cells, new_seg_cells = cell_migration(seg, seg_cells, new_seg_cells, migrate, Q, tau, branch_rule, branch_alpha)

        # 调试信息
        num_cells = new_seg_cells[seg]['num_cells']
        pv_shape = new_seg_cells[seg]['polarity_vectors'].shape
        mi_shape = new_seg_cells[seg]['migration_indicators'].shape
        if num_cells != pv_shape[1]:
            print(f"Warning: Seg {seg} mismatch - num_cells: {num_cells}, polarity_vectors shape: {pv_shape}")
        if num_cells != mi_shape[0]:
            print(f"Warning: Seg {seg} mismatch - num_cells: {num_cells}, migration_indicators shape: {mi_shape}")

    seg_cells = new_seg_cells

    for seg in range(Nseg):
        Ncell[seg] = seg_cells[seg]['num_cells']
        seg_cells[seg]['migration_indicators'] = np.zeros(Ncell[seg], dtype=int)

    for seg in range(Nseg):
        if Ncell[seg] >= 1:
            D[seg] = Ncell[seg] * cell_size / np.pi
        else:
            D[seg] = 0
    G = (np.pi * D**4) / (128 * mu * L)
    H = np.zeros_like(D)
    mask = D > 0
    H[mask] = (32 * mu) / (np.pi * D[mask]**3)
    P, Q, tau = solve_for_flow(G, Pin, Pout, H)

    if t % 20 == 0:
        # 调试信息
        total_cells = sum(seg_cells[seg]['num_cells'] for seg in range(Nseg))
        plot_network(segments, D, P, Q, seg_cells, tau)
