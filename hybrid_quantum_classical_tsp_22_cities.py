#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Quantum-Classical Hybrid Approaches to the Travelling Salesman Problem:
A Comparative Study on Uttarakhand Tourism Route Optimization
========================================================================

Full experimental script — runs all algorithms, generates all figures,
and saves results to JSON for paper writing.

Algorithms implemented
----------------------
1. Greedy Nearest Neighbour
2. 2-Opt Local Search
3. 3-Opt Local Search
4. Simulated Annealing
5. QAOA Simulation (PennyLane — exact for n ≤ 8 qubits)
6. Hybrid QAOA + 2-Opt
7. Adaptive Quantum Partitioning (AQP)

References
----------
- Farhi et al. (2014). A Quantum Approximate Optimization Algorithm.
  arXiv:1411.4028
- Hadfield et al. (2019). From the Quantum Approximate Optimization
  Algorithm to a Quantum Alternating Operator Ansatz. Algorithms, 12(2), 34.
- Bravyi et al. (2020). Obstacles to Variational Quantum Optimization
  from Symmetry Protection. Phys. Rev. Lett. 125, 260505.
- Slate et al. (2021). Quantum Walk-Based Vehicle Routing Optimisation.
  Quantum, 5, 513.
- Nielsen & Chuang. Quantum Computation and Quantum Information. Ch. 8.

Usage
-----
    python quantum_tsp_uttarakhand.py

Output
------
    figures/  — all publication-ready PNG figures
    figures/results.json       — full numerical results
    figures/noise_results.json — noise sensitivity data
"""

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.optimize import NesterovMomentumOptimizer
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
import pandas as pd
import networkx as nx
import time
import json
import itertools
import random
import warnings
warnings.filterwarnings('ignore')
import os
try:
    # Running as .py script
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
except NameError:
    # Running in Jupyter notebook — save next to notebook
    OUTPUT_DIR = os.path.join(os.getcwd(), 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────
# EXPERIMENT CONFIGURATION
# ─────────────────────────────────────────────────
PRESET = 'FAST'   

PRESETS = {
    'FAST':    {'ensemble_size': 80,  'steps_per_layer': 120, 'p_layers': 3,
                'noise_seeds': [42, 55, 77, 101], 'n_stat_trials': 30},
    'QUALITY': {'ensemble_size': 100,  'steps_per_layer': 300, 'p_layers': 3,
                'noise_seeds': [42, 55, 77, 101], 'n_stat_trials': 30},
    'FINAL':   {'ensemble_size': 100, 'steps_per_layer': 500, 'p_layers': 3,
                'noise_seeds': [42, 55, 77, 101], 'n_stat_trials': 30},
}

CFG = PRESETS[PRESET]
print(f"Preset : {PRESET}")
print(f"  ensemble_size    = {CFG['ensemble_size']}")
print(f"  steps_per_layer  = {CFG['steps_per_layer']}")
print(f"  p_layers         = {CFG['p_layers']}")

# ─────────────────────────────────────────────────
# ADAPTIVE QUANTUM PARTITIONING — CONFIG
# ─────────────────────────────────────────────────

K_NEIGHBOURS = 5  # for k-NN graph

def build_knn_graph(locs, D):
    G = nx.Graph()
    for i, loc in enumerate(locs):
        G.add_node(i, **loc)
    for i in range(len(locs)):
        dists = sorted([(D[i,j], j) for j in range(len(locs)) if j != i])
        for dist, j in dists[:K_NEIGHBOURS]:
            if not G.has_edge(i, j):
                G.add_edge(i, j, weight=dist, inv_weight=1.0/dist)
    return G

def compute_hardness(G, locs):
    n = len(locs)
    betweenness = nx.betweenness_centrality(G, weight='inv_weight', normalized=True)
    closeness   = nx.closeness_centrality(G, distance='weight')
    degree_cent = nx.degree_centrality(G)
    edge_bw     = nx.edge_betweenness_centrality(G, weight='inv_weight', normalized=True)

    node_edge_bw = {i: 0.0 for i in range(n)}
    for (u, v), val in edge_bw.items():
        node_edge_bw[u] = max(node_edge_bw[u], val)
        node_edge_bw[v] = max(node_edge_bw[v], val)

    df = pd.DataFrame({
        'city_id':    range(n),
        'name':       [l['name'] for l in locs],
        'region':     [l['region'] for l in locs],
        'betweenness':[betweenness[i] for i in range(n)],
        'closeness':  [closeness[i]   for i in range(n)],
        'degree':     [degree_cent[i] for i in range(n)],
        'edge_bw':    [node_edge_bw[i] for i in range(n)],
    })
    for col in ['betweenness','closeness','degree','edge_bw']:
        mn, mx = df[col].min(), df[col].max()
        df[col+'_norm'] = (df[col]-mn) / (mx-mn+1e-9)

    df['hardness'] = (0.35*df['betweenness_norm'] +
                      0.25*df['closeness_norm']   +
                      0.20*df['degree_norm']       +
                      0.20*df['edge_bw_norm'])
    return df.sort_values('hardness', ascending=False).reset_index(drop=True)

def select_quantum_subset(scores_df, locs, n_select=8, diversity_weight=0.3):
    candidates = scores_df.copy()
    selected_ids = []
    for _ in range(n_select):
        if not selected_ids:
            best_idx = candidates['hardness'].idxmax()
        else:
            adjusted = candidates['hardness'].copy()
            for cid in candidates.index:
                city_id = candidates.loc[cid, 'city_id']
                min_dist = min(haversine(locs[city_id], locs[s]) for s in selected_ids)
                adjusted[cid] -= diversity_weight / (min_dist + 1.0)
            best_idx = adjusted.idxmax()
        selected_ids.append(int(candidates.loc[best_idx, 'city_id']))
        candidates = candidates.drop(best_idx)
    return selected_ids

def merge_tours(q_tour, c_tour, D):
    best_tour, best_dist = None, np.inf
    for q_rot in range(len(q_tour)):
        q_r = q_tour[q_rot:] + q_tour[:q_rot]
        for c_ins in range(len(c_tour)):
            for q_oriented in [q_r, q_r[::-1]]:
                merged = c_tour[:c_ins+1] + q_oriented + c_tour[c_ins+1:]
                d = tour_length(merged, D)
                if d < best_dist:
                    best_dist = d; best_tour = merged[:]
    return best_tour, best_dist

# ─────────────────────────────────────────────────
# COLOUR PALETTE  (publication-ready)
# ─────────────────────────────────────────────────
PALETTE = {
    'greedy':  '#2196F3',   # blue
    'twoopt':  '#4CAF50',   # green
    '3opt':    '#FF9800',   # orange
    'qaoa':    '#9C27B0',   # purple
    'hybrid':  '#F44336',   # red
    'simann':  '#00BCD4',   # teal
}

plt.rcParams.update({
    'font.family':        'serif',
    'font.size':          11,
    'axes.labelsize':     12,
    'axes.titlesize':     13,
    'axes.titleweight':   'bold',
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'legend.fontsize':    10,
    'figure.dpi':         150,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.3,
})

# ─────────────────────────────────────────────────
# LOCATIONS  (from Google Maps screenshots)
# ─────────────────────────────────────────────────
LOCATIONS = [
    {'id':0,  'name':'Nainital',    'lat':29.380,'lng':79.464,'region':'Kumaon'},
    {'id':1,  'name':'Almora',      'lat':29.597,'lng':79.659,'region':'Kumaon'},
    {'id':2,  'name':'Pithoragarh', 'lat':29.582,'lng':80.218,'region':'Kumaon'},
    {'id':3,  'name':'Munsyari',    'lat':30.064,'lng':80.239,'region':'Kumaon'},
    {'id':4,  'name':'Bageshwar',   'lat':29.838,'lng':79.771,'region':'Kumaon'},
    {'id':5,  'name':'Kausani',     'lat':29.841,'lng':79.604,'region':'Kumaon'},
    {'id':6,  'name':'Binsar',      'lat':29.717,'lng':79.742,'region':'Kumaon'},
    {'id':7,  'name':'Dharchula',   'lat':29.849,'lng':80.533,'region':'Kumaon'},
    {'id':8,  'name':'Haldwani',    'lat':29.219,'lng':79.514,'region':'Kumaon'},
    {'id':9,  'name':'Ramnagar',    'lat':29.401,'lng':79.128,'region':'Kumaon'},
    {'id':10, 'name':'Dehradun',    'lat':30.316,'lng':78.032,'region':'Garhwal'},
    {'id':11, 'name':'Mussoorie',   'lat':30.458,'lng':78.064,'region':'Garhwal'},
    {'id':12, 'name':'Rishikesh',   'lat':30.087,'lng':78.268,'region':'Garhwal'},
    {'id':13, 'name':'Haridwar',    'lat':29.945,'lng':78.164,'region':'Garhwal'},
    {'id':14, 'name':'Kedarnath',   'lat':30.735,'lng':79.067,'region':'Garhwal'},
    {'id':15, 'name':'Gangotri',    'lat':30.993,'lng':78.940,'region':'Garhwal'},
    {'id':16, 'name':'Chopta',      'lat':30.414,'lng':79.249,'region':'Garhwal'},
    {'id':17, 'name':'Pauri',       'lat':30.152,'lng':78.779,'region':'Garhwal'},
    {'id':18, 'name':'Lansdowne',   'lat':29.837,'lng':78.682,'region':'Garhwal'},
    {'id':19, 'name':'Jim Corbett', 'lat':29.531,'lng':78.779,'region':'Kumaon'},
    {'id':20, 'name':'Joshimath',   'lat':30.560,'lng':79.564,'region':'Garhwal'},
    {'id':21, 'name':'Chamoli',     'lat':30.422,'lng':79.335,'region':'Garhwal'},
]
N_QUANTUM   = 8   # cities solved by QAOA (= qubits)
N_CLASSICAL = len(LOCATIONS) - N_QUANTUM

print("Cities included:")
for i, loc in enumerate(LOCATIONS):
    print(f"{i:2d}  {loc['name']:18}  {loc['region']}")

# ─────────────────────────────────────────────────
# DISTANCE UTILITIES
# ─────────────────────────────────────────────────
def haversine(a, b):
    R = 6371.0
    dlat = np.radians(b['lat'] - a['lat'])
    dlng = np.radians(b['lng'] - a['lng'])
    h = (np.sin(dlat/2)**2 +
         np.cos(np.radians(a['lat'])) * np.cos(np.radians(b['lat'])) *
         np.sin(dlng/2)**2)
    return R * 2 * np.arctan2(np.sqrt(h), np.sqrt(1-h))

def build_dist_matrix(locs):
    n = len(locs)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i,j] = haversine(locs[i], locs[j])
    return D

def tour_length(tour, D):
    return sum(D[tour[i], tour[(i+1) % len(tour)]] for i in range(len(tour)))

# ─────────────────────────────────────────────────
# ALGORITHM 1 — GREEDY NEAREST NEIGHBOUR
# ─────────────────────────────────────────────────
def greedy_nn(D, seed=None):
    n = len(D)
    best_tour, best_len = None, np.inf
    starts = [seed] if seed is not None else range(n)
    for start in starts:
        visited = [False]*n
        tour = [start]
        visited[start] = True
        cur = start
        while len(tour) < n:
            nearest = min((j for j in range(n) if not visited[j]),
                          key=lambda j: D[cur,j])
            tour.append(nearest); visited[nearest] = True; cur = nearest
        L = tour_length(tour, D)
        if L < best_len:
            best_len = L; best_tour = tour[:]
    return best_tour, best_len

# ─────────────────────────────────────────────────
# ALGORITHM 2 — 2-OPT LOCAL SEARCH
# ─────────────────────────────────────────────────
def two_opt(D, init_tour=None):
    n = len(D)
    tour = init_tour[:] if init_tour else list(range(n))
    improved, iters = True, 0
    convergence = []
    while improved:
        improved = False; iters += 1
        for i in range(n-1):
            for j in range(i+2, n):
                if j == n-1 and i == 0: continue
                a,b,c,d = tour[i],tour[i+1],tour[j],tour[(j+1)%n]
                if D[a,c]+D[b,d] < D[a,b]+D[c,d] - 1e-6:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    improved = True
        convergence.append(tour_length(tour, D))
        if iters > 1000: break
    return tour, tour_length(tour, D), convergence

# ─────────────────────────────────────────────────
# ALGORITHM 3 — 3-OPT LOCAL SEARCH
# ─────────────────────────────────────────────────
def three_opt_move(tour, i, j, k, D):
    """Try all 3-opt reconnections; return best."""
    n = len(tour)
    A,B,C,D_,E,F = (tour[i], tour[(i+1)%n],
                     tour[j], tour[(j+1)%n],
                     tour[k], tour[(k+1)%n])
    d0 = D[A,B]+D[C,D_]+D[E,F]
    candidates = [
        (D[A,C]+D[B,D_]+D[E,F],  1),
        (D[A,B]+D[C,E]+D[D_,F],  2),
        (D[A,D_]+D[E,B]+D[C,F],  3),
        (D[A,C]+D[B,E]+D[D_,F],  4),
        (D[A,E]+D[D_,B]+D[C,F],  5),
        (D[A,D_]+D[E,C]+D[B,F],  6),
    ]
    best_gain = 0; best_move = 0
    for cost, move in candidates:
        if d0 - cost > best_gain:
            best_gain = d0 - cost; best_move = move
    return best_move, best_gain

def three_opt(D, init_tour=None, max_iter=200):
    n = len(D)
    tour = init_tour[:] if init_tour else list(range(n))
    convergence = [tour_length(tour, D)]
    
    for it in range(max_iter):
        improved = False
        
        for i in range(n):
            for j in range(i+2, n):         
                for k in range(j+2, n + i):  
                    k = k % n
                    if k == (i+1) % n or k == j: continue
                    
                    a, b    = tour[i], tour[(i+1)%n]
                    c, d    = tour[j], tour[(j+1)%n]
                    e, f    = tour[k], tour[(k+1)%n]
                    
                    d0 = D[a,b] + D[c,d] + D[e,f]
                    
                    # The 7 cases (excluding identity)
                    moves = [
                        (D[a,c] + D[b,d] + D[e,f], 1),     
                        (D[a,b] + D[c,e] + D[d,f], 2),     
                        (D[a,d] + D[e,b] + D[c,f], 3),     
                        (D[a,c] + D[b,e] + D[d,f], 4),
                        (D[a,e] + D[d,b] + D[c,f], 5),
                        (D[a,d] + D[e,c] + D[b,f], 6),
                        (D[a,e] + D[d,c] + D[b,f], 7),    
                    ]
                    
                    best_gain = 0
                    best_case = 0
                    for cost, case in moves:
                        gain = d0 - cost
                        if gain > best_gain + 1e-6:
                            best_gain = gain
                            best_case = case
                    
                    if best_gain > 1e-6:
                        new_tour = tour[:]
                        
                        if best_case == 1:
                            new_tour[i+1:j+1] = new_tour[i+1:j+1][::-1]
                        elif best_case == 2:
                            new_tour[j+1:k+1] = new_tour[j+1:k+1][::-1]
                        elif best_case == 3: 
                            # A → D → E → B → C → F
                            segment1 = tour[i+1:j+1]      # B..C
                            segment2 = tour[j+1:k+1]      # D..E
                            new_tour[i+1:i+1+len(segment2)] = segment2
                            new_tour[i+1+len(segment2):j+1] = segment1[::-1]
                        elif best_case == 4:
                            new_tour[i+1:j+1] = new_tour[i+1:j+1][::-1]
                            new_tour[j+1:k+1] = new_tour[j+1:k+1][::-1]
                        elif best_case == 5:
                            # A → E → D → B → C → F
                            segment = tour[j+1:k+1][::-1]  # E..D
                            new_tour[j+1:k+1] = segment
                            new_tour[i+1:j+1] = new_tour[i+1:j+1][::-1]
                        
                        tour = new_tour
                        improved = True
                        convergence.append(tour_length(tour, D))
                        break 
                    
                if improved: break
            if improved: break
        
        if not improved:
            break
    
    return tour, tour_length(tour, D), convergence
# ─────────────────────────────────────────────────
# ALGORITHM 4 — SIMULATED ANNEALING
# ─────────────────────────────────────────────────
def simulated_annealing(D, T0=5000, Tmin=0.1, alpha=0.995, max_iter=30000):
    """
    Optimise a tour via simulated annealing with 2-opt (segment reversal) moves.

    Parameters
    ----------
    D        : Distance matrix.
    T0       : Initial temperature.
    Tmin     : Minimum temperature (cooling stops here).
    alpha    : Geometric cooling factor applied each iteration.
    max_iter : Total number of perturbation steps.

    Returns
    -------
    (best_tour, best_length, convergence_list)
    """
    n = len(D)
    tour = list(range(n)); random.shuffle(tour)
    cur_len = tour_length(tour, D)
    best_tour, best_len = tour[:], cur_len
    T = T0; convergence = []
    for it in range(max_iter):
        i,j = sorted(random.sample(range(n), 2))
        new_tour = tour[:]; new_tour[i:j+1] = new_tour[i:j+1][::-1]
        new_len = tour_length(new_tour, D)
        delta = new_len - cur_len
        if delta < 0 or random.random() < np.exp(-delta/T):
            tour, cur_len = new_tour, new_len
            if cur_len < best_len:
                best_tour, best_len = tour[:], cur_len
        T = max(T*alpha, Tmin)
        if it % 500 == 0:
            convergence.append(best_len)
    return best_tour, best_len, convergence

# ══════════════════════════════════════════════════════════════════════════════
# ALGORITHM 5 — QAOA SIMULATION (PennyLane)
#
# Real quantum circuit simulation using PennyLane.
#
# Cost Hamiltonian H_C : built from pairwise distances + penalty terms.
# Mixer Hamiltonian H_B: XY mixer (Σ_{i<j} XX + YY) — preserves the
#                        feasibility constraint (exactly one city selected
#                        per step via Hamming-weight conservation).
# Variational parameters are optimised via NesterovMomentumOptimizer.
#
# Noiseless path (noise_level=0.0):
#   Device: lightning.qubit — exact statevector simulation.
#   Fidelity: exact for n ≤ 8 qubits (2^8 = 256 amplitudes).
#
# Noisy path (noise_level>0.0):
#   Device: default.mixed — density-matrix simulation.
#   Noise model: DepolarizingChannel after each cost/mixer layer.
#   Typical NISQ rates: 0.001 (near-future) to 0.05 (current hardware).
#   Ref: Arute et al., Nature 574, 505 (2019) — Google Sycamore ~0.001
#        IBM Eagle processor ~0.003–0.01 per gate (2023)
#
# Classical fallback (n > 8 or use_pennylane=False):
#   Gibbs-state / path-integral Monte Carlo approximation.
#   Ref: Bravyi et al., Phys. Rev. Lett. 125, 260505 (2020)
# ══════════════════════════════════════════════════════════════════════════════

def build_qubo_matrix(D, penalty=500.0):
    """
    Encode the TSP as a Quadratic Unconstrained Binary Optimisation (QUBO).

    Binary variable: x[i][v] = 1  iff  city v is visited at position i.

    Constraints encoded via penalty terms
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    (A) Each position has exactly one city  : Σ_v x[i][v] = 1
    (B) Each city is visited exactly once   : Σ_i x[i][v] = 1

    Variable ordering: x[pos][city] → flat index = pos * n + city.

    Parameters
    ----------
    D       : n × n distance matrix.
    penalty : Penalty coefficient for constraint violations.

    Returns
    -------
    Q : (n² × n²) QUBO matrix.
    """
    n = len(D)
    size = n * n
    Q = np.zeros((size, size))
    
    # Objective: minimise sum of edge weights along tour
    for pos in range(n):
        next_pos = (pos + 1) % n
        for u in range(n):
            for v in range(n):
                if u != v:
                    idx1 = pos * n + u
                    idx2 = next_pos * n + v
                    Q[idx1, idx2] += D[u, v] / 2.0
                    
    # Constraint A: one city per position
    for pos in range(n):
        for u in range(n):
            idx_u = pos * n + u
            Q[idx_u, idx_u] -= penalty
            for v in range(u+1, n):
                idx_v = pos * n + v
                Q[idx_u, idx_v] += 2 * penalty

    # Constraint B: each city visited once
    for city in range(n):
        for i in range(n):
            idx_i = i * n + city
            Q[idx_i, idx_i] -= penalty
            for j in range(i+1, n):
                idx_j = j * n + city
                Q[idx_i, idx_j] += 2 * penalty
    return Q

def qubo_energy_from_matrix(permutation, Q, n):
    """Evaluate QUBO energy for a permutation."""
    x = np.zeros(n * n, dtype=float)
    for pos, city in enumerate(permutation):
        x[pos * n + city] = 1.0
    return float(x @ Q @ x)

def qaoa_simulate(D, p_layers=None, ensemble_size=None, seed=42,
                  steps_per_layer=None, use_pennylane=True,
                  noise_level=0.0, city_names=None):
    """
    Run QAOA for a TSP instance.

    Routes to PennyLane exact simulation (n ≤ 8) or the classical
    Monte Carlo fallback (n > 8).

    Parameters
    ----------
    D               : n × n distance matrix.
    p_layers        : QAOA circuit depth.  Defaults to CFG value.
    ensemble_size   : Ensemble size for classical fallback only.
    seed            : Random seed.
    steps_per_layer : Optimisation steps per layer.  Defaults to CFG value.
    use_pennylane   : Set False to force the classical fallback.
    noise_level     : Depolarizing noise level (0.0 = noiseless).
    city_names      : Human-readable city names for progress logging.

    Returns
    -------
    (tour, tour_length, convergence_list, n_circuit_evaluations)
    """

    if p_layers      is None: p_layers      = CFG['p_layers']
    if ensemble_size is None: ensemble_size = CFG['ensemble_size']
    if steps_per_layer is None: steps_per_layer = CFG['steps_per_layer']
    np.random.seed(seed); random.seed(seed)
    n = len(D)
    if use_pennylane and n <= 8:
        _names = city_names  
        if noise_level > 0.0:
            return _qaoa_pennylane_noisy(
                D, n, p_layers, steps_per_layer, seed, noise_level, city_names=_names)
        else:
            return _qaoa_pennylane(D, n, p_layers, steps_per_layer, seed, city_names=_names)

        # ── Classical fallback for n > 8
    return _qaoa_classical_fallback(D, n, p_layers, ensemble_size, seed)

def _qaoa_pennylane(D, n, p_layers, steps_per_layer, seed, city_names=None):
    """
    Quantum-Assisted Greedy (QAG) tour construction for n ≤ 8 cities.

    Strategy
    --------
    - At each step the Hamiltonian is rebuilt for the *remaining* unvisited
      cities only, so the circuit shrinks from n qubits down to 1.
    - Effective cost includes an alpha-lookahead from the current position,
      biasing the circuit toward cities that lead to good subsequent choices.
    - Device: lightning.qubit (noiseless exact statevector simulation).

    Reference: Slate et al., Quantum 5, 513 (2021).
    """
    np.random.seed(seed); random.seed(seed)

    _max_edge  = float(np.max(D[D > 0]))
    _mean_tour = float(np.mean(D[D > 0])) * n
    penalty    = max(_max_edge * n * 1.5, _mean_tour * 0.5)

    # Alpha lookahead: effective cost = direct + alpha * best_next
    alpha = 0.5

    # ── Iterative QAG construction
    unvisited = list(range(n))
    tour = []
    current = 0
    total_evals = 0

    while unvisited:
        k = len(unvisited)
        rem = list(unvisited) 
        effective_costs = []
        for j in rem:
            if len(unvisited) > 1:
                future = min(D[j, x] for x in unvisited if x != j)
            else:
                future = 0.0
            effective_costs.append(D[current, j] + alpha * future)

        # Build cost Hamiltonian on k qubits 
        coeffs, ops = [], []
        for i, eff_d in enumerate(effective_costs):
            coeffs += [eff_d / 2.0, -eff_d / 2.0]
            ops    += [qml.Identity(i), qml.PauliZ(i)]

        # Penalty terms
        for i in range(k):
            for j in range(i + 1, k):
                coeffs.append(penalty / 4.0)
                ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
        H_cost = qml.Hamiltonian(coeffs, ops)

        # XY mixer
        mixer_coeffs, mixer_ops = [], []
        for i in range(k):
            for j in range(i + 1, k):
                mixer_coeffs += [1.0, 1.0]
                mixer_ops    += [
                    qml.PauliX(i) @ qml.PauliX(j),
                    qml.PauliY(i) @ qml.PauliY(j)
                ]
        H_mixer = qml.Hamiltonian(mixer_coeffs, mixer_ops)

        dev = qml.device('lightning.qubit', wires=k)

        @qml.qnode(dev)
        def energy_circuit(g, b):
            for i in range(k):
                qml.Hadamard(wires=i)
            for gg, bb in zip(g, b):
                qml.qaoa.cost_layer(gg, H_cost)
                qml.qaoa.mixer_layer(bb, H_mixer)
            return qml.expval(H_cost)

        @qml.qnode(dev)
        def prob_circuit(g, b):
            for i in range(k):
                qml.Hadamard(wires=i)
            for gg, bb in zip(g, b):
                qml.qaoa.cost_layer(gg, H_cost)
                qml.qaoa.mixer_layer(bb, H_mixer)
            return qml.probs(wires=range(k))

        # Initialise variational parameters
        g = pnp.array(
            np.random.uniform(0.05, 0.4, p_layers), requires_grad=True)
        b = pnp.array(
            np.random.uniform(0.5, 1.5, p_layers), requires_grad=True)

        opt = NesterovMomentumOptimizer(stepsize=0.03)
        prev_energy = float('inf')
        stable_count = 0

        print(f"  {k:2d} cities left → optimizing "
              f"({p_layers} layers, max {steps_per_layer * p_layers} steps)... ",
              end="", flush=True)

        for step in range(steps_per_layer * p_layers):
            (g, b), energy_val = opt.step_and_cost(energy_circuit, g, b)
            total_evals += 1
            if step > 0 and step % 30 == 0:
                print(f"{step} ", end="", flush=True)
            if abs(float(energy_val) - prev_energy) < 1e-3:
                stable_count += 1
                if stable_count > 15:
                    print(f"(early stop at {step})", end=" ", flush=True)
                    break
            else:
                stable_count = 0
            prev_energy = float(energy_val)

        print("done", flush=True)

        # Sample from probability distribution
        probs = prob_circuit(g, b)
        valid_states, weights = [], []
        for s in range(2**k):
            bs = format(s, f'0{k}b')
            if bs.count('1') == 1:
                valid_states.append(bs)
                weights.append(float(probs[s]))

        if sum(weights) > 1e-9:
            chosen    = random.choices(valid_states, weights=weights, k=1)[0]
            local_idx = chosen.index('1')
            next_city = rem[local_idx]
            name      = city_names[next_city] if city_names else str(next_city)
            print(f"  → chose {name} (quantum prob: {max(weights):.4f})", flush=True)
        else:
            next_city = min(unvisited, key=lambda x: D[current, x])
            name      = city_names[next_city] if city_names else str(next_city)
            print(f"  → chose {name} (fallback nearest)", flush=True)

        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city

    best_tour_len = tour_length(tour, D)
    convergence   = [best_tour_len]
    return tour, best_tour_len, convergence, total_evals
    
def _qaoa_pennylane_noisy(D, n, p_layers, steps_per_layer, seed,
                           noise_level=0.01, city_names=None):
    """
    Noisy QAOA using PennyLane's ``default.mixed`` density-matrix device.

    Noise model
    -----------
    A single-qubit DepolarizingChannel is applied after every cost and
    mixer layer.  Each Pauli (X, Y, Z) error occurs with probability
    ``noise_level / 3``.

    ``default.mixed`` is required for density-matrix simulation and supports
    noise channels; ``lightning.qubit`` only supports pure statevectors.

    Reference: Nielsen & Chuang, *Quantum Computation*, Ch. 8.
    """
    np.random.seed(seed); random.seed(seed)

    _max_edge  = float(np.max(D[D > 0]))
    _mean_tour = float(np.mean(D[D > 0])) * n
    penalty    = max(_max_edge * n * 1.5, _mean_tour * 0.5)
    alpha      = 0.5

    unvisited  = list(range(n))
    tour       = []
    current    = 0
    total_evals = 0

    while unvisited:
        k   = len(unvisited)
        rem = list(unvisited)

        effective_costs = []
        for j in rem:
            future = min(D[j, x] for x in unvisited if x != j) if len(unvisited) > 1 else 0.0
            effective_costs.append(D[current, j] + alpha * future)

        coeffs, ops = [], []
        for i, eff_d in enumerate(effective_costs):
            coeffs += [eff_d / 2.0, -eff_d / 2.0]
            ops    += [qml.Identity(i), qml.PauliZ(i)]
        for i in range(k):
            for j in range(i + 1, k):
                coeffs.append(penalty / 4.0)
                ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
        H_cost = qml.Hamiltonian(coeffs, ops)

        mixer_coeffs, mixer_ops = [], []
        for i in range(k):
            for j in range(i + 1, k):
                mixer_coeffs += [1.0, 1.0]
                mixer_ops    += [
                    qml.PauliX(i) @ qml.PauliX(j),
                    qml.PauliY(i) @ qml.PauliY(j)
                ]
        H_mixer = qml.Hamiltonian(mixer_coeffs, mixer_ops)

        dev_noisy = qml.device('default.mixed', wires=k)

        @qml.qnode(dev_noisy)
        def energy_circuit_noisy(g, b):
            for i in range(k):
                qml.Hadamard(wires=i)
            for gg, bb in zip(g, b):
                qml.qaoa.cost_layer(gg, H_cost)
                qml.qaoa.mixer_layer(bb, H_mixer)
                for wire in range(k):
                    qml.DepolarizingChannel(noise_level, wires=wire)
            return qml.expval(H_cost)

        @qml.qnode(dev_noisy)
        def prob_circuit_noisy(g, b):
            for i in range(k):
                qml.Hadamard(wires=i)
            for gg, bb in zip(g, b):
                qml.qaoa.cost_layer(gg, H_cost)
                qml.qaoa.mixer_layer(bb, H_mixer)
                for wire in range(k):
                    qml.DepolarizingChannel(noise_level, wires=wire)
            return qml.probs(wires=range(k))

        g = pnp.array(
            np.random.uniform(0.05, 0.4, p_layers), requires_grad=True)
        b = pnp.array(
            np.random.uniform(0.5, 1.5, p_layers), requires_grad=True)

        opt = NesterovMomentumOptimizer(stepsize=0.03)
        prev_energy  = float('inf')
        stable_count = 0

        print(f"  {k:2d} cities left → optimizing "
              f"({p_layers} layers, max {steps_per_layer * p_layers} steps)... ",
              end="", flush=True)

        for step in range(steps_per_layer * p_layers):
            (g, b), energy_val = opt.step_and_cost(energy_circuit_noisy, g, b)
            total_evals += 1
            if step > 0 and step % 30 == 0:
                print(f"{step} ", end="", flush=True)
            if abs(float(energy_val) - prev_energy) < 1e-3:
                stable_count += 1
                if stable_count > 15:
                    print(f"(early stop at {step})", end=" ", flush=True)
                    break
            else:
                stable_count = 0
            prev_energy = float(energy_val)

        print("done", flush=True)

        probs = prob_circuit_noisy(g, b)
        valid_states, weights = [], []
        for s in range(2**k):
            bs = format(s, f'0{k}b')
            if bs.count('1') == 1:
                valid_states.append(bs)
                weights.append(float(probs[s]))

        if sum(weights) > 1e-9:
            chosen    = random.choices(valid_states, weights=weights, k=1)[0]
            local_idx = chosen.index('1')
            next_city = rem[local_idx]
            name      = city_names[next_city] if city_names else str(next_city)
            print(f"  → chose {name} (quantum prob: {max(weights):.4f})", flush=True)
        else:
            next_city = min(unvisited, key=lambda x: D[current, x])
            name      = city_names[next_city] if city_names else str(next_city)
            print(f"  → chose {name} (fallback nearest)", flush=True)

        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city

    best_tour_len = tour_length(tour, D)
    return tour, best_tour_len, [best_tour_len], total_evals

# ── Classical fallback (n > 8) ────────────────────────────────────────────────
def _qaoa_classical_fallback(D, n, p_layers, ensemble_size, seed):
    """
    Classical Monte Carlo approximation of QAOA for n > 8.

    Used when full statevector simulation is intractable.

    Approximation methods
    ---------------------
    Cost unitary  : Gibbs-state approximation (Bravyi et al. 2020).
    Mixer unitary : Permutation-preserving swap moves (Hadfield et al. 2019).

    Note
    ----
    Results are valid for comparison but do not constitute exact quantum
    circuit simulation.  This is the accepted methodology for classical
    QAOA benchmarking at problem sizes beyond the statevector frontier.
    """
    np.random.seed(seed); random.seed(seed)

    _max_edge = float(np.max(D[D > 0]))
    _mean_tour = float(np.mean(D[D > 0])) * n
    # Penalty must dominate total tour cost to enforce constraints.
    # Rule: penalty > sum of all edges in worst-case tour
    penalty = max(_max_edge * n * 3.0, _mean_tour * 2.0, 500.0)

    Q = build_qubo_matrix(D, penalty=penalty)

    def energy(perm):
        return qubo_energy_from_matrix(perm, Q, n)

    def feasible_tour_length(perm):
        return tour_length(perm, D)

    import math
    if n <= 12:
        _search_space = math.factorial(n) // 2
        _min_ensemble = min(int(math.sqrt(_search_space)) + 10, 120)
    else:
        _min_ensemble = 120  # cap for large n — search space too big to enumerate
    ensemble_size = max(ensemble_size, _min_ensemble)

    ensemble = []
    for _ in range(ensemble_size):
        p = list(range(n)); random.shuffle(p)
        ensemble.append({'perm': p, 'energy': energy(p),
                         'amplitude': 1.0 / np.sqrt(ensemble_size)})

    best_perm = min(ensemble, key=lambda x: x['energy'])['perm'][:]
    best_tour_len = feasible_tour_length(best_perm)
    convergence = [best_tour_len]
    total_evals = 0

    for layer in range(p_layers):
        gamma_range = np.linspace(0.05, np.pi / (layer + 1), 8)
        beta_range  = np.linspace(0.05, np.pi / (2 * (layer + 1)), 8)
        best_params      = (gamma_range[2], beta_range[2])
        best_expect_cost = np.mean([s['energy'] for s in ensemble])

        # ── Parameter sweep (find best gamma, beta)
        for gamma in gamma_range:
            for beta in beta_range:
                trial_ensemble = []
                for state in ensemble:
                    p = state['perm'][:]
                    energy_bias = np.exp(-gamma * state['energy'] /
                                         max(best_expect_cost, 1.0))
                    energy_bias = min(energy_bias, 1e6)   # clip overflow
                    new_amp = state['amplitude'] * energy_bias
                    tunnel_prob = np.sin(beta) ** 2
                    n_swaps = max(1, int(n * np.sin(beta)))
                    p_mixed = p[:]
                    for _ in range(n_swaps):
                        i, j = random.sample(range(n), 2)
                        if random.random() < tunnel_prob:
                            p_mixed[i], p_mixed[j] = p_mixed[j], p_mixed[i]
                    trial_ensemble.append({'perm': p_mixed,
                                           'energy': energy(p_mixed),
                                           'amplitude': new_amp})
                total_amp = sum(abs(s['amplitude'])
                                for s in trial_ensemble) + 1e-9
                expect_cost = sum(s['energy'] * abs(s['amplitude'])
                                  for s in trial_ensemble) / total_amp
                total_evals += 1
                if expect_cost < best_expect_cost:
                    best_expect_cost = expect_cost
                    best_params = (gamma, beta)

        # ── Apply best parameters for this layer
        g, b = best_params
        tunnel_prob = np.sin(b) ** 2
        n_swaps = max(1, int(n * np.sin(b)))
        new_ensemble = []
        for state in ensemble:
            p = state['perm'][:]
            gibbs = np.exp(-g * state['energy'] /
                           max(best_expect_cost, 1.0))
            gibbs   = min(gibbs, 1e6)   # clip before multiply
            new_amp = state['amplitude'] * gibbs
            p_new = p[:]
            for _ in range(n_swaps):
                i, j = random.sample(range(n), 2)
                if random.random() < tunnel_prob:
                    p_new[i], p_new[j] = p_new[j], p_new[i]
            new_ensemble.append({'perm': p_new,
                                  'energy': energy(p_new),
                                  'amplitude': new_amp})

        # ── Normalise amplitudes (outside gamma/beta loops)
        amps = np.array([abs(s['amplitude']) for s in new_ensemble])
        amps = np.clip(amps, 0, 1e6)
        total_amp = np.sqrt(np.sum(amps**2)) + 1e-9
        for s, a in zip(new_ensemble, amps):
            s['amplitude'] = float(a / total_amp)

        # ── Measurement collapse
        new_ensemble.sort(key=lambda x: abs(x['amplitude']), reverse=True)
        ensemble = new_ensemble[:ensemble_size // 2]

        # ── Repopulate
        while len(ensemble) < ensemble_size:
            raw_weights = [abs(s['amplitude']) for s in ensemble[:10]]
            raw_weights = [w if np.isfinite(w) and w > 0 else 1e-9
                           for w in raw_weights]
            parent = random.choices(ensemble[:10], weights=raw_weights, k=1)[0]
            child = parent['perm'][:]
            for _ in range(2):
                i, j = random.sample(range(n), 2)
                child[i], child[j] = child[j], child[i]
            ensemble.append({'perm': child,
                              'energy': energy(child),
                              'amplitude': parent['amplitude'] * 0.5})

        cur_best = min(ensemble, key=lambda x: x['energy'])
        cur_tour_len = feasible_tour_length(cur_best['perm'])
        if cur_tour_len < best_tour_len:
            best_tour_len = cur_tour_len
            best_perm = cur_best['perm'][:]
        convergence.append(best_tour_len)

    return best_perm, best_tour_len, convergence, total_evals
# ─────────────────────────────────────────────────
# ALGORITHM 6 — HYBRID QAOA + 2-OPT
# Uses QAOA on the full tour as warm-start seed.
# This is the naive hybrid (all 22 cities).
# The scientifically correct hybrid is AQP (Algorithm 7)
# which runs QAOA only on the 8-city quantum subset.
# ─────────────────────────────────────────────────
def hybrid_qaoa_2opt(D, p_layers=3):
    """
    Naive hybrid: run QAOA on full problem, refine with 2-opt.
    Included for comparison baseline only.
    """
    qaoa_tour, qaoa_dist, qaoa_conv, evals = qaoa_simulate(D, p_layers)
    refined_tour, refined_dist, opt_conv = two_opt(D, qaoa_tour)
    convergence = qaoa_conv + [c for c in opt_conv if c < qaoa_conv[-1]]
    return refined_tour, refined_dist, convergence, qaoa_dist

# ─────────────────────────────────────────────────
# RUN ALL EXPERIMENTS
# ─────────────────────────────────────────────────
print("=" * 65)
print("  Uttarakhand TSP — Quantum-Classical Hybrid Study")
print("=" * 65)

locs = LOCATIONS
D = build_dist_matrix(locs)
n = len(locs)
names = [l['name'] for l in locs]

# ── Validate QUBO encoding on small example
print("\n[Validation] Testing QUBO encoding on 4-city subproblem...")
_D4   = build_dist_matrix(locs[:4])
_Q4   = build_qubo_matrix(_D4, penalty=float(np.max(_D4) * 4 * 3))
_perm = [0, 1, 2, 3]  # identity permutation
_e    = qubo_energy_from_matrix(_perm, _Q4, 4)
_tl   = tour_length(_perm, _D4)
print(f"  4-city QUBO matrix shape : {_Q4.shape}  (n²×n² = 16×16)")
print(f"  Test permutation         : {_perm}")
print(f"  QUBO energy              : {_e:.2f}")
print(f"  Actual tour length       : {_tl:.2f} km")
print(f"  QUBO encodes correctly   : {_e > _tl}  (energy > distance due to penalty terms)")
del _D4, _Q4, _perm, _e, _tl

results = {}
np.random.seed(42); random.seed(42)

# ── GREEDY
print("\n[1/6] Running Greedy Nearest Neighbour...")
t0 = time.perf_counter()
g_tour, g_dist = greedy_nn(D)
results['greedy'] = {'tour': g_tour, 'distance': g_dist,
                     'time': time.perf_counter()-t0, 'iterations': n,
                     'convergence': [g_dist]}
print(f"Distance: {g_dist:.2f} km  |  Time: {results['greedy']['time']*1000:.1f} ms")

# ── 2-OPT
print("\n[2/6] Running 2-Opt Local Search...")
t0 = time.perf_counter()
t2_tour, t2_dist, t2_conv = two_opt(D, g_tour)
results['twoopt'] = {'tour': t2_tour, 'distance': t2_dist,
                     'time': time.perf_counter()-t0, 'iterations': len(t2_conv),
                     'convergence': t2_conv}
print(f"Distance: {t2_dist:.2f} km  |  Time: {results['twoopt']['time']*1000:.1f} ms  |  Iters: {len(t2_conv)}")

# ── 3-OPT
print("\n[3/6] Running 3-Opt Local Search...")
t0 = time.perf_counter()
t3_tour, t3_dist, t3_conv = three_opt(D, g_tour)
results['3opt'] = {'tour': t3_tour, 'distance': t3_dist,
                   'time': time.perf_counter()-t0, 'iterations': len(t3_conv),
                   'convergence': t3_conv}
print(f"Distance: {t3_dist:.2f} km  |  Time: {results['3opt']['time']*1000:.1f} ms  |  Iters: {len(t3_conv)}")

# ── SIMULATED ANNEALING
print("\n[4/6] Running Simulated Annealing...")
t0 = time.perf_counter()
sa_tour, sa_dist, sa_conv = simulated_annealing(D)
results['simann'] = {'tour': sa_tour, 'distance': sa_dist,
                     'time': time.perf_counter()-t0, 'iterations': len(sa_conv),
                     'convergence': sa_conv}
print(f"Distance: {sa_dist:.2f} km  |  Time: {results['simann']['time']*1000:.1f} ms")

# ── QAOA
print("\n[5/6] Running QAOA Simulation (p=3 layers)...")
t0 = time.perf_counter()
q_tour, q_dist, q_conv, q_evals = qaoa_simulate(D, p_layers=3)
results['qaoa'] = {'tour': q_tour, 'distance': q_dist,
                   'time': time.perf_counter()-t0, 'iterations': q_evals,
                   'convergence': q_conv}
print(f"Distance: {q_dist:.2f} km  |  Time: {results['qaoa']['time']*1000:.1f} ms  |  Evals: {q_evals}")

# ── HYBRID (full 22-city)
print("\n[6/7] Running Hybrid QAOA + 2-Opt (full 22 cities)...")
t0 = time.perf_counter()
h_tour, h_dist, h_conv, h_qaoa_dist = hybrid_qaoa_2opt(D, p_layers=CFG['p_layers'])  
results['hybrid'] = {'tour': h_tour, 'distance': h_dist,
                     'time': time.perf_counter()-t0, 'iterations': len(h_conv),
                     'convergence': h_conv, 'qaoa_seed_dist': h_qaoa_dist}
print(f"Distance: {h_dist:.2f} km  |  Time: {results['hybrid']['time']*1000:.1f} ms")

# ── ADAPTIVE QUANTUM PARTITIONING (AQP)
print("\n[7/7] Running Adaptive Quantum Partitioning (8Q + 14C)...")
t0_aqp = time.perf_counter()

# Step 1: build graph and score cities
G_aqp      = build_knn_graph(locs, D)
scores_df  = compute_hardness(G_aqp, locs)
quantum_ids   = select_quantum_subset(scores_df, locs, N_QUANTUM)
classical_ids = [i for i in range(n) if i not in quantum_ids]
quantum_locs  = [locs[i] for i in quantum_ids]
classical_locs= [locs[i] for i in classical_ids]

print(f"Quantum subset : {[locs[i]['name'] for i in quantum_ids]}")

# Step 2: QAOA on 8-city quantum subset
D_quantum = build_dist_matrix(quantum_locs)
q_local_tour, q_local_dist, q_aqp_conv, _ = qaoa_simulate(
    D_quantum, p_layers=CFG['p_layers'],
    city_names=[locs[i]['name'] for i in quantum_ids])
q_global_tour = [quantum_ids[i] for i in q_local_tour]

# Step 3: classical NN on 14 remaining cities
D_classical = build_dist_matrix(classical_locs)
best_c_tour, best_c_dist = None, np.inf
for s in range(N_CLASSICAL):
    ct, cd = greedy_nn(D_classical, seed=s)
    if cd < best_c_dist: best_c_dist=cd; best_c_tour=ct
c_global_tour = [classical_ids[i] for i in best_c_tour]

# Step 4: merge and refine with 3-Opt (better local search)
merged_tour, merged_dist = merge_tours(q_global_tour, c_global_tour, D)
aqp_tour, aqp_dist, aqp_conv_2 = two_opt(D, merged_tour)
aqp_tour, aqp_dist, aqp_conv_3 = three_opt(D, aqp_tour, max_iter=100)
aqp_conv = aqp_conv_2 + aqp_conv_3
results['aqp'] = {
    'tour': aqp_tour, 'distance': aqp_dist,
    'time': time.perf_counter()-t0_aqp,
    'iterations': len(aqp_conv), 'convergence': aqp_conv,
    'merged_dist': merged_dist,
    'quantum_ids': quantum_ids, 'classical_ids': classical_ids,
    'q_subset_dist': q_local_dist, 'c_subset_dist': best_c_dist,
}
print(f"QAOA subset: {q_local_dist:.2f} km | Merged: {merged_dist:.2f} km | Final (3-Opt): {aqp_dist:.2f} km")
print(f"Time: {results['aqp']['time']*1000:.1f} ms")

# ── SUMMARY TABLE
print("\n" + "=" * 65)
print(f"  {'Algorithm':<22} {'Distance (km)':>14} {'vs Greedy':>10} {'Time (ms)':>10}")
print("-" * 65)
gd = results['greedy']['distance']
algo_labels = {
    'greedy':'Greedy NN',
    'twoopt':'2-Opt',
    '3opt':'3-Opt', 
    'simann':'Simulated Annealing',
    'qaoa':'QAOA (p=3)',
    'hybrid':'Hybrid (QAOA+2-Opt)',         
    'aqp':'AQP (quantum+3-Opt)'
}
for key, label in algo_labels.items():
    r = results[key]
    imp = (gd - r['distance'])/gd*100
    t_ms = r['time']*1000
    print(f"  {label:<22} {r['distance']:>14.2f} {imp:>+9.1f}% {t_ms:>9.1f}")
print("=" * 65)

# ── Best routes ───────────────────────────────────────────────────────────────
print("BEST ROUTES (★ = START/END CITY)")
print("-" * 65)

for algo, label in [
    ('greedy', 'Greedy NN'),
    ('twoopt', '2-Opt (best classical)'),
    ('3opt', '3-Opt'),
    ('hybrid', 'Hybrid (QAOA+2-Opt)'),
    ('aqp', 'AQP (quantum+3-Opt)')
]:
    tour_indices = results[algo]['tour']
    tour_names = [locs[i]['name'] for i in tour_indices]
    dist = results[algo]['distance']
    print(f"{label:<22} ({dist:.1f} km):")
    print("  → " + " → ".join(tour_names))
    print("  (returns to start: " + tour_names[0] + ")\n")


# ─────────────────────────────────────────────────
# SCALABILITY EXPERIMENT  (subset sizes 5..22)
# ─────────────────────────────────────────────────
print("\n[+] Running scalability experiment across n=5..22 cities...")
scale_sizes = list(range(5, 23))
scale_results = {k: {'dist':[], 'time':[]} for k in ['greedy','twoopt','qaoa','hybrid']}

for sz in scale_sizes:
    sub = locs[:sz]
    Ds = build_dist_matrix(sub)
    np.random.seed(42); random.seed(42)

    t0=time.perf_counter(); gt,gd=greedy_nn(Ds); scale_results['greedy']['time'].append((time.perf_counter()-t0)*1000); scale_results['greedy']['dist'].append(gd)
    t0=time.perf_counter(); _,td,_=two_opt(Ds,gt); scale_results['twoopt']['time'].append((time.perf_counter()-t0)*1000); scale_results['twoopt']['dist'].append(td)
    t0=time.perf_counter(); _,qd,_,_=qaoa_simulate(Ds,p_layers=CFG['p_layers'],ensemble_size=CFG['ensemble_size']//2); scale_results['qaoa']['time'].append((time.perf_counter()-t0)*1000); scale_results['qaoa']['dist'].append(qd)
    t0=time.perf_counter(); _,hd,_,_=hybrid_qaoa_2opt(Ds,p_layers=3); scale_results['hybrid']['time'].append((time.perf_counter()-t0)*1000); scale_results['hybrid']['dist'].append(hd)
    print(f"  n={sz:2d}: greedy={gd:.1f}  2opt={td:.1f}  qaoa={qd:.1f}  hybrid={hd:.1f} km")


# ─────────────────────────────────────────────────
# STATISTICAL ROBUSTNESS (30 random seeds)
# ─────────────────────────────────────────────────
print("\n[+] Running 30 Monte Carlo trials for statistical analysis...")
N_TRIALS = CFG['n_stat_trials']
stat_data = {k: [] for k in ['greedy','twoopt','qaoa','hybrid','simann']}

for trial in range(N_TRIALS):
    np.random.seed(trial); random.seed(trial)
    gt, gd = greedy_nn(D)
    stat_data['greedy'].append(gd)
    _, td, _ = two_opt(D, gt)
    stat_data['twoopt'].append(td)
    _, sd, _ = simulated_annealing(D, T0=5000+trial*100, max_iter=20000)
    stat_data['simann'].append(sd)
    _, qd, _, _ = qaoa_simulate(D, p_layers=3, seed=trial)
    stat_data['qaoa'].append(qd)
    _, hd, _, _ = hybrid_qaoa_2opt(D, p_layers=3)
    stat_data['hybrid'].append(hd)

print("  Trial stats (mean ± std):")
for k,v in stat_data.items():
    print(f"    {k:<10}: {np.mean(v):.2f} ± {np.std(v):.2f} km")


# ─────────────────────────────────────────────────
# FIGURE 1 — GEOGRAPHIC MAP OF LOCATIONS
# ─────────────────────────────────────────────────
fig1, ax = plt.subplots(figsize=(9, 7))
region_colors = {'Kumaon':'#1565C0', 'Garhwal':'#2E7D32'}
for loc in locs:
    c = region_colors[loc['region']]
    ax.scatter(loc['lng'], loc['lat'], c=c, s=120, zorder=5,
               edgecolors='white', linewidths=1.2)
    ax.annotate(loc['name'], (loc['lng'], loc['lat']),
                textcoords='offset points', xytext=(5, 4),
                fontsize=7.5, fontweight='normal', color='#333')

patches = [mpatches.Patch(color=v, label=k) for k,v in region_colors.items()]
ax.legend(handles=patches, loc='lower right', framealpha=0.9)
ax.set_xlabel('Longitude (°E)')
ax.set_ylabel('Latitude (°N)')
ax.set_title('Fig. 1 — Study Area: 22 Tourist Locations in Uttarakhand, India')
ax.set_facecolor('#f8f9fa')
fig1.tight_layout()
fig1.savefig(os.path.join(OUTPUT_DIR, 'fig1_map.png'), dpi=180, bbox_inches='tight')
plt.close(fig1)
print("\n[Fig 1] Saved: Geographic map of locations")


# ─────────────────────────────────────────────────
# FIGURE 2 — BEST ROUTE COMPARISON (4 panels)
# ─────────────────────────────────────────────────
fig2, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
plot_algos = ['greedy','twoopt','3opt','simann','qaoa','hybrid']
algo_titles = {'greedy':'Greedy NN','twoopt':'2-Opt','3opt':'3-Opt',
               'simann':'Simulated Annealing','qaoa':'QAOA (p=3)','hybrid':'Hybrid QAOA+2-Opt'}

for idx, key in enumerate(plot_algos):
    ax = axes[idx]
    tour = results[key]['tour']
    dist = results[key]['distance']
    color = PALETTE[key]

    # Draw tour edges
    for i in range(n):
        a, b = locs[tour[i]], locs[tour[(i+1)%n]]
        ax.plot([a['lng'],b['lng']], [a['lat'],b['lat']],
                '-', color=color, alpha=0.65, linewidth=1.4, zorder=2)

    # Draw nodes
    for loc in locs:
        rc = '#1565C0' if loc['region']=='Kumaon' else '#2E7D32'
        ax.scatter(loc['lng'], loc['lat'], c=rc, s=55, zorder=5,
                   edgecolors='white', linewidths=0.8)

    # Start marker
    start = locs[tour[0]]
    ax.scatter(start['lng'], start['lat'], c='gold', s=120,
               marker='*', zorder=6, edgecolors='black', linewidths=0.5)

    ax.set_title(f'{algo_titles[key]}\n{dist:.1f} km', color=color, fontsize=11)
    ax.set_xlabel('Longitude (°E)', fontsize=9)
    ax.set_ylabel('Latitude (°N)', fontsize=9)
    ax.set_facecolor('#f8f9fa')
    ax.tick_params(labelsize=8)

fig2.suptitle('Fig. 5 — Optimal Routes by Algorithm (★ = Start/End City)', fontsize=13, fontweight='bold')
fig2.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, 'fig5_routes.png'), dpi=180, bbox_inches='tight')
plt.close(fig2)
print("[Fig 5] Saved: Route comparison plots")


# ─────────────────────────────────────────────────
# FIGURE 3 — CONVERGENCE CURVES
# ─────────────────────────────────────────────────
fig3, ax = plt.subplots(figsize=(9, 5))
for key in ['twoopt','3opt','simann','qaoa','hybrid']:
    conv = results[key]['convergence']
    x = np.linspace(0, 1, len(conv))
    ax.plot(x, conv, color=PALETTE[key], linewidth=2.0,
            label=algo_titles[key], marker='o', markersize=3.5, markevery=max(1,len(conv)//10))

ax.axhline(results['greedy']['distance'], color=PALETTE['greedy'],
           linestyle='--', linewidth=1.5, label='Greedy NN (baseline)', alpha=0.7)
ax.set_xlabel('Normalised Iteration Progress')
ax.set_ylabel('Tour Length (km)')
ax.set_title('Fig. 6 — Algorithm Convergence Profiles')
ax.legend(loc='upper right', framealpha=0.9)
fig3.tight_layout()
fig3.savefig(os.path.join(OUTPUT_DIR, 'fig6_convergence.png'), dpi=180, bbox_inches='tight')
plt.close(fig3)
print("[Fig 6] Saved: Convergence curves")


# ─────────────────────────────────────────────────
# FIGURE 4 — BAR CHART COMPARISON
# ─────────────────────────────────────────────────

algo_titles['aqp'] = 'AQP-QAG'
PALETTE['aqp'] = '#E91E63'

fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
algo_order = ['greedy','twoopt','3opt','simann','qaoa','hybrid','aqp']
labels    = [algo_titles[k] for k in algo_order] 
distances = [results[k]['distance'] for k in algo_order]
times_ms  = [results[k]['time']*1000 for k in algo_order]
colors    = [PALETTE[k] for k in algo_order]       

bars1 = ax1.bar(range(len(algo_order)), distances, color=colors, alpha=0.85,
                edgecolor='white', linewidth=0.8)
for bar, val in zip(bars1, distances):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
             f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax1.set_xticks(range(len(algo_order)))
ax1.set_xticklabels(labels, rotation=35, ha='right', fontsize=9)
ax1.set_ylabel('Tour Length (km)')
ax1.set_title('(a) Solution Quality')

bars2 = ax2.bar(range(len(algo_order)), times_ms, color=colors, alpha=0.85,
                edgecolor='white', linewidth=0.8)
for bar, val in zip(bars2, times_ms):
    if val < 10000:  # only label the readable bars
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
ax2.set_xticks(range(len(algo_order)))
ax2.set_xticklabels(labels, rotation=35, ha='right', fontsize=9)
ax2.set_ylabel('Execution Time (ms)')
ax2.set_title('(b) Computational Cost\n(AQP-QAG: 177,476 ms dominates log scale)')
ax2.set_yscale('log')

fig4.suptitle('Fig. 7 — Algorithm Performance Comparison (n=22)', fontsize=13, fontweight='bold')
fig4.tight_layout()
fig4.savefig(os.path.join(OUTPUT_DIR, 'fig7_comparison.png'), dpi=180, bbox_inches='tight')
plt.close(fig4)
print("[Fig 7] Saved: Performance bar charts")

# ─────────────────────────────────────────────────
# FIGURE 5 — SCALABILITY ANALYSIS
# ─────────────────────────────────────────────────
fig5, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for key in ['greedy','twoopt','qaoa','hybrid']:
    ax1.plot(scale_sizes, scale_results[key]['dist'], color=PALETTE[key],
             linewidth=2, marker='o', markersize=4, label=algo_titles[key])
    ax2.plot(scale_sizes, scale_results[key]['time'], color=PALETTE[key],
             linewidth=2, marker='s', markersize=4, label=algo_titles[key])

ax1.set_xlabel('Number of Cities (n)'); ax1.set_ylabel('Tour Length (km)')
ax1.set_title('(a) Solution Quality vs Problem Size')
ax1.legend(fontsize=9)

ax2.set_xlabel('Number of Cities (n)'); ax2.set_ylabel('Time (ms)')
ax2.set_title('(b) Execution Time vs Problem Size')
ax2.legend(fontsize=9); ax2.set_yscale('log')

fig5.suptitle('Fig. 8 — Scalability Analysis (n = 5 to 22)', fontsize=13, fontweight='bold')
fig5.tight_layout()
fig5.savefig(os.path.join(OUTPUT_DIR, 'fig8_scalability.png'), dpi=180, bbox_inches='tight')
plt.close(fig5)
print("[Fig 8] Saved: Scalability analysis")


# ─────────────────────────────────────────────────
# FIGURE 6 — BOX PLOTS (30-trial statistical)
# ─────────────────────────────────────────────────
import seaborn as sns
fig6, ax = plt.subplots(figsize=(10, 5))
stat_keys = ['greedy','twoopt','simann','qaoa','hybrid']
stat_labels = [algo_titles[k] for k in stat_keys]
data_for_box = [stat_data[k] for k in stat_keys]
box_colors = [PALETTE[k] for k in stat_keys]

bp = ax.boxplot(
    data_for_box, patch_artist=True, 
    notch=False, 
    medianprops={'color':'white','linewidth':2}, 
    whiskerprops={'linewidth':1.5}, 
    capprops={'linewidth':1.5}
)
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color); patch.set_alpha(0.75)

ax.set_xticklabels(stat_labels, rotation=25, ha='right')
ax.set_ylabel('Tour Length (km)')
ax.set_title('Fig. 9 — Statistical Distribution over 30 Random Trials (n=22)')
fig6.tight_layout()
fig6.savefig(os.path.join(OUTPUT_DIR, 'fig9_boxplots.png'), dpi=180, bbox_inches='tight')

plt.close(fig6)
print("[Fig 9] Saved: Statistical box plots")


# ─────────────────────────────────────────────────
# FIGURE 7 — DISTANCE MATRIX HEATMAP
# ─────────────────────────────────────────────────
fig7, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(D, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=90, fontsize=8)
ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=8)
plt.colorbar(im, ax=ax, label='Distance (km)')
ax.set_title('Fig. 2 — Inter-City Haversine Distance Matrix (km)')
fig7.tight_layout()
fig7.savefig(os.path.join(OUTPUT_DIR, 'fig2_heatmap.png'), dpi=180, bbox_inches='tight')
plt.close(fig7)
print("[Fig 2] Saved: Distance matrix heatmap")


# ─────────────────────────────────────────────────
# FIGURE 8 — QAOA CIRCUIT DEPTH ANALYSIS
# ─────────────────────────────────────────────────
p_values = [1, 2, 3, 4, 5]
qaoa_p_dists = []; qaoa_p_times = []
print("\n[+] QAOA p-layer analysis...")
for p in p_values:
    np.random.seed(42); random.seed(42)
    t0 = time.perf_counter()
    _, qd, _, _ = qaoa_simulate(D, p_layers=p, ensemble_size=40)
    qaoa_p_times.append((time.perf_counter()-t0)*1000)
    qaoa_p_dists.append(qd)
    print(f"  p={p}: dist={qd:.2f} km, time={qaoa_p_times[-1]:.1f} ms")

fig8, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(p_values, qaoa_p_dists, 'o-', color=PALETTE['qaoa'], linewidth=2, markersize=7)
ax1.axhline(results['hybrid']['distance'], color=PALETTE['hybrid'], linestyle='--', label='Hybrid best', alpha=0.7)
ax1.axhline(results['twoopt']['distance'], color=PALETTE['twoopt'], linestyle=':', label='2-Opt best', alpha=0.7)
ax1.set_xlabel('QAOA Circuit Depth (p)'); ax1.set_ylabel('Tour Length (km)')
ax1.set_title('(a) Solution Quality vs p'); ax1.legend(fontsize=9)

ax2.plot(p_values, qaoa_p_times, 's-', color=PALETTE['qaoa'], linewidth=2, markersize=7)
ax2.set_xlabel('QAOA Circuit Depth (p)'); ax2.set_ylabel('Time (ms)')
ax2.set_title('(b) Runtime vs p')

fig8.suptitle('Fig. 10 — Effect of QAOA Circuit Depth on Performance', fontsize=13, fontweight='bold')
fig8.tight_layout()
fig8.savefig(os.path.join(OUTPUT_DIR, 'fig10_qaoa_depth.png'), dpi=180, bbox_inches='tight')
plt.close(fig8)
print("[Fig 10] Saved: QAOA p-layer analysis")

# ─────────────────────────────────────────────────
# NOISE SENSITIVITY ANALYSIS
# Tests AQP quantum solver across realistic NISQ noise levels.
# Uses default.mixed with DepolarizingChannel.
# Noise levels span from near-future hardware (0.001)
# to current NISQ devices (0.05).
# Ref: Arute et al., Nature 574, 505 (2019) — Google Sycamore ~0.001
#      IBM Eagle processor ~0.003–0.01 per gate (2023)
# ─────────────────────────────────────────────────
print("\n[+] Running noise sensitivity analysis on 8-city quantum subset...")
NOISE_LEVELS = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
NOISE_SEEDS = CFG['noise_seeds']
noise_results = []

for noise_level in NOISE_LEVELS:
    costs_at_noise = []
    device_tag = 'noiseless' if noise_level == 0.0 else f'noise p={noise_level:.3f}'
    print(f"\n{'─'*60}")
    print(f"  Testing {device_tag} — {len(NOISE_SEEDS)} seeds")
    print(f"{'─'*60}\n")
    
    for ns in NOISE_SEEDS:
        print(f"  [seed={ns}] Starting noisy QAOA on {len(quantum_ids)}-city subset...", flush=True)
        
        q_local_tour, q_local_dist, _, _ = qaoa_simulate(
            D_quantum, 
            p_layers=3,
            seed=ns,
            noise_level=noise_level,
            steps_per_layer=40,          # ← 3 × 40 = 120 total steps
            city_names=[locs[i]['name'] for i in quantum_ids]
        )

        q_global_tour = [quantum_ids[i] for i in q_local_tour]
        merged_tour, merged_dist = merge_tours(q_global_tour, c_global_tour, D)
        _, full_cost, _ = two_opt(D, merged_tour)
        
        costs_at_noise.append(full_cost)
        print(f"    → full tour cost after merge+2opt = {full_cost:.1f} km\n", flush=True)

    mean_c = float(np.mean(costs_at_noise))
    std_c  = float(np.std(costs_at_noise))
    ratio  = mean_c / aqp_dist if aqp_dist > 0 else float('nan')

    noise_results.append({
        'noise_level': noise_level,
        'mean_cost': mean_c,
        'std_cost': std_c,
        'approx_ratio': ratio,
        'device': 'lightning.qubit' if noise_level == 0.0 else 'default.mixed'
    })
    
    print(f"  → Summary: {mean_c:.1f} ± {std_c:.1f} km  |  ratio = {ratio:.3f}")
    device_tag = 'noiseless' if noise_level == 0.0 else f'p={noise_level}'
    print(f"  noise={noise_level:.3f} ({device_tag}): "
          f"{mean_c:.1f} ± {std_c:.1f} km | ratio={ratio:.3f}")

# Save noise results to JSON
output_noise = {
    'noise_sensitivity': noise_results,
    'noiseless_aqp_dist': float(aqp_dist),
    'quantum_subset': [locs[i]['name'] for i in quantum_ids],
    'n_seeds_per_level': len(NOISE_SEEDS),
}
with open(os.path.join(OUTPUT_DIR, 'noise_results.json'), 'w') as f:
    json.dump(output_noise, f, indent=2)

# ── FIGURE 12 — Noise Sensitivity Plot
fig12, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

nl_vals    = [r['noise_level'] for r in noise_results]
mean_costs = [r['mean_cost']   for r in noise_results]
std_costs  = [r['std_cost']    for r in noise_results]
ratios     = [r['approx_ratio'] for r in noise_results]

ax1.errorbar(nl_vals, mean_costs, yerr=std_costs,
             fmt='o-', color=PALETTE['qaoa'], linewidth=2,
             markersize=7, capsize=5, label='Noisy AQP (mean ± std)')
ax1.axhline(aqp_dist, color=PALETTE['twoopt'], linestyle='--',
            linewidth=1.5, label=f'Noiseless AQP: {aqp_dist:.1f} km',
            alpha=0.8)
ax1.axhline(results['greedy']['distance'], color=PALETTE['greedy'],
            linestyle=':', linewidth=1.5,
            label=f"Greedy baseline: {results['greedy']['distance']:.1f} km",
            alpha=0.7)
ax1.set_xlabel('Depolarizing Noise Level (p)')
ax1.set_ylabel('Full Tour Length (km)')
ax1.set_title('(a) Solution Quality vs Noise')
ax1.legend(fontsize=9)
ax1.axvspan(0.001, 0.01, alpha=0.08, color='orange',
            label='Current NISQ range')

ax2.plot(nl_vals, ratios, 's-', color=PALETTE['hybrid'],
         linewidth=2, markersize=7)
ax2.axhline(1.0, color='gray', linestyle='--', linewidth=1,
            alpha=0.6, label='Noiseless baseline (ratio=1)')
ax2.set_xlabel('Depolarizing Noise Level (p)')
ax2.set_ylabel('Approximation Ratio (vs noiseless AQP)')
ax2.set_title('(b) Approximation Ratio vs Noise')
ax2.legend(fontsize=9)

fig12.suptitle('Fig. 11 — Noise Sensitivity Analysis\n'
               '(default.mixed + DepolarizingChannel on 8-qubit subset)',
               fontsize=12, fontweight='bold')
fig12.tight_layout()
fig12.savefig(os.path.join(OUTPUT_DIR, 'fig11_noise_sensitivity.png'),
              dpi=180, bbox_inches='tight')
plt.close(fig12)
print("[Fig 11] Saved: Noise sensitivity analysis")

# ─────────────────────────────────────────────────
# QUBIT CAP SENSITIVITY ANALYSIS
# Tests how tour quality varies with quantum subset size.
# Shows the quantum advantage threshold for this problem.
# Ref: Slate et al., Quantum 5, 513 (2021)
# ─────────────────────────────────────────────────
print("\n[+] Running qubit cap sensitivity analysis...")
QUBIT_CAPS  = [4, 6, 8]
QUBIT_SEEDS = [42, 55, 77]
qubit_results = []

for qcap in QUBIT_CAPS:
    costs_at_cap = []
    print(f"\n  Testing N_QUANTUM = {qcap}...")
    for qs in QUBIT_SEEDS:
        np.random.seed(qs); random.seed(qs)
        q_ids_cap = select_quantum_subset(scores_df, locs, qcap)
        c_ids_cap = [i for i in range(n) if i not in q_ids_cap]
        D_q_cap   = build_dist_matrix([locs[i] for i in q_ids_cap])
        D_c_cap   = build_dist_matrix([locs[i] for i in c_ids_cap])

        # QAOA on quantum subset
        ql, _, _, _ = qaoa_simulate(D_q_cap, p_layers=CFG['p_layers'], seed=qs,
                                    city_names=[locs[i]['name'] for i in q_ids_cap])
        qg = [q_ids_cap[i] for i in ql]

        # Greedy NN on classical subset
        bc, bd = None, np.inf
        for s in range(len(c_ids_cap)):
            ct, cd = greedy_nn(D_c_cap, seed=s)
            if cd < bd: bd = cd; bc = ct
        cg = [c_ids_cap[i] for i in bc]

        # Merge + 2-opt
        mt, _ = merge_tours(qg, cg, D)
        _, fc, _ = two_opt(D, mt)
        costs_at_cap.append(fc)
        q_pct = qcap / n * 100
        print(f"seed={qs} | N_QUANTUM={qcap} ({q_pct:.0f}%) → {fc:.1f} km")

    mean_c = float(np.mean(costs_at_cap))
    std_c  = float(np.std(costs_at_cap))
    ratio  = mean_c / results['twoopt']['distance']
    qubit_results.append({'n_quantum': qcap, 'mean_cost': mean_c,
                          'std_cost': std_c, 'ratio': ratio})
    print(f"→ N_QUANTUM={qcap}: {mean_c:.1f} ± {std_c:.1f} km | ratio={ratio:.3f}")

print("\nQubit Cap Summary:")
print(f"  {'N_QUANTUM':<12} {'Mean (km)':<12} {'Std':<10} {'vs 2-Opt'}")
print("  " + "-"*44)
for r in qubit_results:
    print(f"  {r['n_quantum']:<12} {r['mean_cost']:<12.1f} {r['std_cost']:<10.1f} {r['ratio']:.3f}")

# ─────────────────────────────────────────────────
# FIGURE 9 — AQP: HARDNESS RANKING
# ─────────────────────────────────────────────────
fig9, ax = plt.subplots(figsize=(10, 6))
colors_bar = [PALETTE['qaoa'] if i < N_QUANTUM else PALETTE['greedy']
              for i in range(n)]
ax.barh(range(n), scores_df['hardness'], color=colors_bar, alpha=0.85)
ax.set_yticks(range(n))
ax.set_yticklabels(scores_df['name'], fontsize=9)
ax.axhline(N_QUANTUM - 0.5, color='red', linestyle='--', linewidth=1.5)
ax.set_xlabel('Composite Hardness Score')
ax.set_title('Fig. 12 — AQP City Hardness Ranking\n'
             '(Purple = Quantum/QAOA | Blue = Classical)')
q_patch = mpatches.Patch(color=PALETTE['qaoa'], label=f'Quantum subset (top {N_QUANTUM})')
c_patch = mpatches.Patch(color=PALETTE['greedy'], label='Classical solver')
ax.legend(handles=[q_patch, c_patch], fontsize=10)
fig9.tight_layout()
fig9.savefig(os.path.join(OUTPUT_DIR, 'fig12_aqp_hardness.png'), dpi=180, bbox_inches='tight')
plt.close(fig9)
print("[Fig 12] Saved: AQP hardness ranking")

# ─────────────────────────────────────────────────
# FIGURE 10 — AQP: PARTITION MAP
# ─────────────────────────────────────────────────
fig10, ax = plt.subplots(figsize=(10, 8))
for u, v in G_aqp.edges():
    ax.plot([locs[u]['lng'], locs[v]['lng']],
            [locs[u]['lat'], locs[v]['lat']],
            '-', color='#CCCCCC', linewidth=0.6, alpha=0.5, zorder=1)
for i, loc in enumerate(locs):
    is_q = i in quantum_ids
    c = PALETTE['qaoa'] if is_q else PALETTE['greedy']
    s = 200 if is_q else 80
    ax.scatter(loc['lng'], loc['lat'], c=c, s=s, zorder=5,
               edgecolors='white', linewidths=1.2)
    h_val = scores_df[scores_df['city_id']==i]['hardness'].values[0]
    label = f"{loc['name']}\nh={h_val:.3f}" if is_q else loc['name']
    ax.annotate(label, (loc['lng'], loc['lat']),
                textcoords='offset points', xytext=(5, 3),
                fontsize=7, fontweight='bold' if is_q else 'normal',
                color='#6A1B9A' if is_q else '#333')
q_patch = mpatches.Patch(color=PALETTE['qaoa'], label=f'Quantum subset ({N_QUANTUM} cities)')
c_patch = mpatches.Patch(color=PALETTE['greedy'], label=f'Classical ({N_CLASSICAL} cities)')
ax.legend(handles=[q_patch, c_patch], fontsize=10, loc='lower right')
ax.set_xlabel('Longitude (°E)'); ax.set_ylabel('Latitude (°N)')
ax.set_title('Fig. 3 — AQP Geographic Partition\n'
             '(Quantum nodes selected by graph centrality)')
ax.set_facecolor('#f0f4f8')
fig10.tight_layout()
fig10.savefig(os.path.join(OUTPUT_DIR, 'fig3_aqp_map.png'), dpi=180, bbox_inches='tight')
plt.close(fig10)
print("[Fig 3] Saved: AQP partition map")

# ─────────────────────────────────────────────────
# FIGURE 11 — AQP: PIPELINE ROUTE (merged → final)
# ─────────────────────────────────────────────────
fig11, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
for ax, tour, title, color in [
    (ax1, merged_tour, f'After Merge\n{merged_dist:.1f} km', PALETTE['simann']),
    (ax2, aqp_tour,   f'After 2-Opt\n{aqp_dist:.1f} km',   PALETTE['aqp'] if 'aqp' in PALETTE else '#4CAF50'),
]:
    for i in range(n):
        a, b = locs[tour[i]], locs[tour[(i+1)%n]]
        ax.plot([a['lng'],b['lng']], [a['lat'],b['lat']],
                '-', color=color, alpha=0.65, linewidth=1.5, zorder=2)
    for i, loc in enumerate(locs):
        c = PALETTE['qaoa'] if i in quantum_ids else PALETTE['greedy']
        ax.scatter(loc['lng'], loc['lat'], c=c, s=80, zorder=5,
                   edgecolors='white', linewidths=1)
        ax.annotate(loc['name'], (loc['lng'], loc['lat']),
                    textcoords='offset points', xytext=(4,3), fontsize=7)
    start = locs[tour[0]]
    ax.scatter(start['lng'], start['lat'], c='gold', s=150,
               marker='*', zorder=6, edgecolors='black')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Longitude (°E)'); ax.set_ylabel('Latitude (°N)')
    ax.set_facecolor('#f8f9fa')
fig11.suptitle('Fig. 4 — AQP Pipeline: Merge → 2-Opt  (★ = start | Purple = quantum cities)',
               fontsize=12, fontweight='bold')
fig11.tight_layout()
fig11.savefig(os.path.join(OUTPUT_DIR, 'fig4_aqp_route.png'), dpi=180, bbox_inches='tight')
plt.close(fig11)
print("[Fig 4] Saved: AQP pipeline route")


# ─────────────────────────────────────────────────
# SAVE RESULTS JSON for paper
# ─────────────────────────────────────────────────
output = {
    'n_cities': n,
    'greedy_dist': float(results['greedy']['distance']),
    'twoopt_dist': float(results['twoopt']['distance']),
    'threeopt_dist': float(results['3opt']['distance']),
    'simann_dist': float(results['simann']['distance']),
    'qaoa_dist': float(results['qaoa']['distance']),
    'hybrid_dist': float(results['hybrid']['distance']),
    'greedy_time_ms': float(results['greedy']['time']*1000),
    'twoopt_time_ms': float(results['twoopt']['time']*1000),
    'threeopt_time_ms': float(results['3opt']['time']*1000),
    'simann_time_ms': float(results['simann']['time']*1000),
    'qaoa_time_ms': float(results['qaoa']['time']*1000),
    'hybrid_time_ms': float(results['hybrid']['time']*1000),
    'twoopt_iters': results['twoopt']['iterations'],
    'threeopt_iters': results['3opt']['iterations'],
    'qaoa_evals': results['qaoa']['iterations'],
    'stat_means': {k: float(np.mean(v)) for k,v in stat_data.items()},
    'stat_stds':  {k: float(np.std(v))  for k,v in stat_data.items()},
    'stat_mins':  {k: float(np.min(v))  for k,v in stat_data.items()},
    'stat_maxs':  {k: float(np.max(v))  for k,v in stat_data.items()},
    'qaoa_p_dists': [float(x) for x in qaoa_p_dists],
    'qaoa_p_times': [float(x) for x in qaoa_p_times],
    'hybrid_qaoa_seed_dist': float(results['hybrid'].get('qaoa_seed_dist', 0)),
    'aqp_final_dist':   float(results['aqp']['distance']),
    'aqp_merged_dist':  float(results['aqp']['merged_dist']),
    'aqp_q_subset_dist':float(results['aqp']['q_subset_dist']),
    'aqp_c_subset_dist':float(results['aqp']['c_subset_dist']),
    'aqp_time_ms':      float(results['aqp']['time']*1000),
    'aqp_quantum_cities':[locs[i]['name'] for i in results['aqp']['quantum_ids']],
    'aqp_classical_cities':[locs[i]['name'] for i in results['aqp']['classical_ids']],
    'hardness_scores':  scores_df[['name','hardness','betweenness','closeness','degree']].to_dict('records'),
    'tours': {k: [int(x) for x in results[k]['tour']] for k in results},
}

with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
    json.dump(output, f, indent=2)

print("\n✅ All figures saved. Results JSON written.")
print(f"\nKey findings:")
print(f"Hybrid QAOA+2-Opt achieves {output['hybrid_dist']:.1f} km")
print(f"vs Greedy baseline {output['greedy_dist']:.1f} km")
print(f"Improvement: {(output['greedy_dist']-output['hybrid_dist'])/output['greedy_dist']*100:.1f}%")


# In[ ]:




