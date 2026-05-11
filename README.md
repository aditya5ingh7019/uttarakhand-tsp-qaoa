# Quantum-Classical Hybrid TSP: Uttarakhand Tourism Route Optimisation

Research code for a comparative study of quantum-classical hybrid TSP approaches applied to 22 tourist destinations across Uttarakhand, India. Developed at Amity University Uttar Pradesh, Lucknow.

## Paper

**Adaptive Quantum Partitioning for Hybrid QAOA-Based TSP Optimisation: A Case Study on Uttarakhand Tourism Routing**
Aditya Singh, Rajiv Pandey, Pooja Srivastava
Department of Physics / Department of Computer Science & Engineering, Amity University Uttar Pradesh

## Requirements

- Python 3.10+
- PennyLane 0.38+
- PennyLane-Lightning 0.38+
- NumPy 1.24+
- NetworkX 3.1+
- Pandas 2.0+
- Matplotlib 3.7+
- Seaborn 0.12+

## Installation

```bash
pip install pennylane pennylane-lightning numpy networkx pandas matplotlib seaborn
```

## Running

```bash
python uttarakhand_tsp.py
```

All outputs are saved to the directory defined by `CUSTOM_OUTPUT_DIR` at the top of the script (default: `C:\Users\Aditya Singh\uttarakhand_multi_regime_outputs`). Change this path before running on a different machine.

## Output Files

| File | Description |
|------|-------------|
| `results.json` | Full numerical results: distances, times, tours, stats, multi-regime, hardness scores |
| `noise_results.json` | Noise sensitivity data across 6 depolarising error levels |
| `fig1_map.png` | Geographic distribution of 22 destinations |
| `fig2_heatmap.png` | Inter-city Haversine distance matrix heatmap |
| `fig3_aqp_map.png` | AQP geographic partition — quantum vs classical city assignment |
| `fig4_aqp_route.png` | AQP pipeline: merged tour (1204 km) → refined tour (883 km) |
| `fig5_routes.png` | Best routes found by each of the 6 non-AQP algorithms |
| `fig6_convergence.png` | Algorithm convergence profiles (normalised iterations) |
| `fig7_comparison.png` | Solution quality and runtime bar charts |
| `fig8_scalability.png` | Scalability analysis across n = 5 to 22 cities |
| `fig9_boxplots.png` | Statistical distribution over 30 Monte Carlo trials |
| `fig10_qaoa_depth.png` | Effect of QAOA circuit depth p = 1 to 5 |
| `fig11_noise_sensitivity.png` | Noise sensitivity under depolarising error (6 levels, 4 seeds each) |
| `fig12_aqp_hardness.png` | Composite hardness scores for all 22 cities |

## Algorithms

Seven algorithms are implemented and benchmarked:

| # | Algorithm | Type |
|---|-----------|------|
| 1 | Greedy Nearest Neighbour | Classical |
| 2 | 2-Opt Local Search | Classical |
| 3 | 3-Opt Local Search | Classical |
| 4 | Simulated Annealing | Classical |
| 5 | QAOA Monte Carlo Surrogate | Classical fallback (n > 8) |
| 6 | Hybrid QAOA + 2-Opt | Classical surrogate + refinement |
| 7 | **AQP-QAG** (Adaptive Quantum Partitioning with Quantum-Assisted Greedy) | **Genuine quantum** |

Only Algorithm 7 uses real quantum circuit simulation. Algorithms 5 and 6 use a classical Gibbs-state Monte Carlo approximation for n > 8, which is explicitly labelled as a surrogate in all results.

## AQP-QAG Framework

The Adaptive Quantum Partitioning pipeline runs in six steps:

1. **Hardness scoring** — builds a k-NN graph (k = 5) over all 22 cities and computes composite centrality: betweenness (0.35) + closeness (0.25) + degree (0.20) + edge betweenness (0.20)
2. **Quantum subset selection** — selects the top-8 hardest cities with a diversity penalty (weight = 0.3) to avoid geographic clustering
3. **QAG circuit** — runs iterative QAOA on the 8-city quantum subset using PennyLane (`lightning.qubit`), with XY mixer and alpha-lookahead cost (alpha = 0.5)
4. **Classical sub-tour** — solves the remaining 14 cities with best-start Greedy NN
5. **Exhaustive merge** — evaluates all 16 × 14 = 224 merge candidates and selects the minimum-distance combination
6. **Local search refinement** — applies 2-Opt then 3-Opt (max 100 iterations) to the merged tour

Quantum cities selected (consistent across runs): Kausani, Pauri, Lansdowne, Chamoli, Ramnagar, Almora, Joshimath, Nainital.

## Quantum Backends

| Backend | Used for | Notes |
|---------|----------|-------|
| `lightning.qubit` | Noiseless AQP runs, multi-regime, qubit cap analysis | Exact statevector, 2^k amplitudes |
| `default.mixed` | Noise sensitivity analysis | Density matrix + DepolarizingChannel |

Exact statevector simulation is feasible for k ≤ 8 qubits (256 amplitudes, ~4 KB). For n > 8, the code automatically falls back to the classical Monte Carlo surrogate.

## Multi-Regime Validation

Two smaller instances are included to validate the quantum solver independently of the partition step:

- **4-city** (Nainital, Almora, Bageshwar, Kausani) — approximation ratio 1.000
- **8-city** (Nainital, Almora, Pithoragarh, Bageshwar, Kausani, Dehradun, Rishikesh, Haridwar) — approximation ratio 1.000

Both achieve the 2-Opt optimum exactly, confirming quantum circuit quality within the 8-qubit boundary.

## Noise Sensitivity

The AQP quantum subset is tested under single-qubit depolarising noise at six error rates:

```
η ∈ {0.0, 0.001, 0.005, 0.010, 0.020, 0.050}
```

4 seeds per level: {42, 55, 77, 101}. Noise runs use a reduced optimisation budget of 120 total gradient steps (vs 360 in the main run) to manage the cost of density-matrix simulation. Results are saved to `noise_results.json`.

## Qubit Cap Sensitivity

The AQP pipeline is repeated for N_QUANTUM ∈ {4, 6, 8} to measure the effect of hardware capacity on solution quality. 3 seeds per configuration: {42, 55, 77}.

## Presets

Change `PRESET` at the top of the script:

| Preset | steps/layer | Max total steps | Ensemble size | Recommended use |
|--------|-------------|-----------------|---------------|-----------------|
| `FAST` | 120 | 360 | 80 | Paper results, rapid iteration |
| `QUALITY` | 300 | 900 | 100 | Stronger convergence verification |
| `FINAL` | 500 | 1500 | 100 | Publication-grade final runs |

## Expected Runtime (FAST preset, standard CPU, no GPU)

| Experiment | Approximate time |
|------------|-----------------|
| Main algorithms (1–7) | ~3.5 minutes |
| Multi-regime validation | ~7 minutes |
| Scalability (n = 5–22) | ~10 minutes |
| Statistical robustness (30 trials) | ~8 minutes |
| QAOA depth analysis (p = 1–5) | ~2 minutes |
| Noise sensitivity (6 levels × 4 seeds) | ~35–40 minutes |
| Qubit cap analysis (3 caps × 3 seeds) | ~15 minutes |
| **Total** | **~75–85 minutes** |

The noise sensitivity analysis dominates total runtime because `default.mixed` density-matrix simulation is substantially slower than `lightning.qubit` statevector simulation.

## Reproducibility

All results are fully reproducible with fixed seeds. Global seed = 42. Results are written to `results.json` and `noise_results.json` immediately after each experiment section completes, so partial results are preserved if the run is interrupted.

## Key Results (FAST preset, p = 3)

| Algorithm | Distance (km) | vs 2-Opt | Time (ms) | Quantum? |
|-----------|---------------|----------|-----------|----------|
| Greedy NN | 820.84 | +1.7% | 1.9 | No |
| 2-Opt | 806.84 | baseline | 0.4 | No |
| 3-Opt | 806.84 | 0.0% | 11.5 | No |
| Simulated Annealing | 828.55 | +2.7% | 386.2 | No |
| QAOA Surrogate | 1653.60 | +104.9% | 1743.7 | No |
| QAOA Surrogate + 2-Opt | 828.55 | +2.7% | 1652.2 | No |
| **AQP-QAG (proposed)** | **882.95** | **+9.4%** | **195,183** | **Yes** |

At n ≤ 8 (multi-regime), genuine QAOA achieves approximation ratio 1.000. The 9.4% gap at n = 22 is attributable to the merge step between independently constructed sub-tours, not to quantum circuit quality.

## References

- Farhi et al. (2014). A Quantum Approximate Optimization Algorithm. arXiv:1411.4028
- Hadfield et al. (2019). From QAOA to a Quantum Alternating Operator Ansatz. Algorithms 12(2), 34.
- Slate et al. (2021). Quantum Walk-Based Vehicle Routing Optimisation. Quantum 5, 513.
- Bravyi et al. (2020). Obstacles to Variational Quantum Optimization. PRL 125, 260505.
- Nielsen & Chuang. Quantum Computation and Quantum Information. Ch. 8.

## Citation

If you use this code, please cite the associated paper (details to be added upon publication).

## License

For academic use. Contact the authors for other uses.
