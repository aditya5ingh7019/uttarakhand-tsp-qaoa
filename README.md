# Adaptive Quantum Partitioning for Hybrid QAOA-Based TSP Optimisation
### A Case Study on Uttarakhand Tourism Routing

**Authors:** Aditya Singh, Rajiv Pandey, Pooja Srivastava  
**Affiliation:** Amity University Uttar Pradesh, Lucknow  
**Journal:** Soft Computing (Springer, Q2) — Under Review  

---

## Overview

This repository contains the complete experimental code, pre-computed results, and all publication figures for the paper *"Adaptive Quantum Partitioning for Hybrid QAOA-Based TSP Optimisation: A Case Study on Uttarakhand Tourism Routing"*.

We benchmark seven algorithms on a 22-city Travelling Salesman Problem (TSP) constructed from real tourist locations in Uttarakhand, India. Our proposed method — **Adaptive Quantum Partitioning with Quantum-Assisted Greedy (AQP-QAG)** — uses a graph-hardness metric to select the 8 most routing-critical cities for quantum treatment via QAOA (PennyLane), solving the remaining 14 cities classically, then merging and post-optimising the tours.

---

## Key Results (22-City Uttarakhand TSP)

| Algorithm | Tour Length (km) | Time (ms) |
|---|---|---|
| Greedy Nearest Neighbour | 820.84 | 3.9 |
| 2-Opt | 806.84 | 0.8 |
| 3-Opt | 806.84 | 14.7 |
| Simulated Annealing | 828.55 | 546.5 |
| QAOA Monte Carlo Surrogate | 1,653.60 | 2,092.2 |
| Hybrid QAOA + 2-Opt | 828.55 | 2,564.9 |
| **AQP-QAG (proposed)** | **882.95** | **460,732** |

- AQP merged tour before post-optimisation: 1,135.19 km → final 882.95 km (**22.2% improvement**)  
- Statistical robustness (30 trials): SA mean 841.97 ± 27.74 km, QAOA mean 1,777.79 ± 65.40 km  
- QAOA circuit depth p=5 best distance: **504.69 km** on isolated 8-city quantum subset  
- Noise sensitivity: only 26.7 km spread across 50× noise increase — confirms NISQ robustness  

---

## Repository Structure

```
uttarakhand-tsp-qaoa/
│
├── hybrid_quantum_classical_tsp_22_cities.ipynb   ← Full experimental notebook
├── hybrid_quantum_classical_tsp_22_cities.py      ← Python script export of the notebook
│
├── results.json                                   ← Pre-computed QUALITY preset results
├── qaoa_p5_results.json                           ← Verified p=5 circuit depth run
│
├── README.md
└── NOTEBOOK_GUIDE.md                              ← Cell-by-cell guide to the notebook
```

---

## Notebook Guide (Important — Read Before Running)

The notebook (`hybrid_quantum_classical_tsp_22_cities.ipynb`) has **two distinct phases**. They use different presets and serve different purposes.

### Phase 1: Cells 1–6 — Exploratory Run (FAST preset)
- **Preset:** `FAST` (ensemble_size=80, steps_per_layer=120, p_layers=3)
- **Purpose:** Quick sanity check and parameter exploration during development
- **NOT the source of any paper results**
- Runtime: ~3–5 minutes

### Phase 2: Cells 7 onward — Paper Results (QUALITY preset)
- **Preset:** `QUALITY` (ensemble_size=100, steps_per_layer=300, p_layers=3)
- **Purpose:** All tables and figures in the paper come from this phase
- **This is the paper's experimental source**
- Runtime: ~8–10 hours on standard CPU

**To reproduce paper results, start execution from Cell 7.**

See `NOTEBOOK_GUIDE.md` for a complete cell-by-cell breakdown.

---

## Pre-Computed Results

If you do not want to re-run the full ~8-hour experiment, all paper results are available in the JSON files:

**`results.json`** — Main QUALITY preset output:
```json
{
  "greedy_dist": 820.84,   "greedy_time_ms": 3.9,
  "twoopt_dist": 806.84,   "twoopt_time_ms": 0.8,
  "threeopt_dist": 806.84, "threeopt_time_ms": 14.7,
  "simann_dist": 828.55,   "simann_time_ms": 546.5,
  "qaoa_dist": 1653.60,    "qaoa_time_ms": 2092.2,
  "hybrid_dist": 828.55,   "hybrid_time_ms": 2564.9,
  "aqp_final_dist": 882.95,"aqp_time_ms": 460732.18,
  ...
}
```

**`qaoa_p5_results.json`** — Real p=5 QAOA run on 8-city quantum subset:
```json
{
  "dist": 504.69,
  "time_ms": 681221,
  "evaluations": 5267
}
```

> **Note:** `results.json` contains a `qaoa_p_dists` field with placeholder values for p=5 (480.0 km, 550,000 ms) that were manually set during early exploration and are **not used in the paper**. All paper Table 12 and Figure 10 values come from `qaoa_p5_results.json`.

---

## Quantum Subset Selected by AQP

The graph-hardness metric selected these 8 cities for QAOA treatment:

**Quantum (8 cities):** Kausani, Pauri, Lansdowne, Chamoli, Ramnagar, Almora, Joshimath, Nainital  
**Classical (14 cities):** Pithoragarh, Munsyari, Bageshwar, Binsar, Dharchula, Haldwani, Dehradun, Mussoorie, Rishikesh, Haridwar, Kedarnath, Gangotri, Chopta, Jim Corbett

---

## Requirements

```
pennylane>=0.35.0
pennylane-lightning
numpy
scipy
matplotlib
pandas
networkx
```

Install with:
```bash
pip install pennylane pennylane-lightning numpy scipy matplotlib pandas networkx
```

Python 3.9 or higher recommended.

---

## Running the Experiment

### Option A — Load pre-computed results (instant)
Open the notebook and run only the figure-regeneration cells (Cell 7 section onward, skipping the long algorithm cells). All results load from `results.json` and `qaoa_p5_results.json`.

### Option B — Full reproduction from scratch (~8–10 hours)
```bash
# In the notebook, run from Cell 7 onward
# OR run the Python script (currently set to FAST preset by default):
python hybrid_quantum_classical_tsp_22_cities.py
```

> **Warning:** The `.py` file as uploaded runs the **FAST preset** by default (line 81: `PRESET = 'FAST'`). To reproduce paper results, change line 81 to `PRESET = 'QUALITY'` before running. Also update `CUSTOM_OUTPUT_DIR` on line 71 to a path on your machine.

---

## City Coordinates

All 22 cities use real geographic coordinates (WGS84). Distances are computed using the **Haversine formula** (great-circle distances). The dataset covers both Kumaon and Garhwal divisions of Uttarakhand.

---

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{singh2025aqp,
  title   = {Adaptive Quantum Partitioning for Hybrid QAOA-Based TSP Optimisation:
             A Case Study on Uttarakhand Tourism Routing},
  author  = {Singh, Aditya and Pandey, Rajiv and Srivastava, Pooja},
  journal = {Soft Computing},
  year    = {2025},
  note    = {Under Review}
}
```

---

## References

- Farhi, E., Goldstone, J., & Gutmann, S. (2014). A Quantum Approximate Optimization Algorithm. *arXiv:1411.4028*
- Hadfield, S. et al. (2019). From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz. *Algorithms, 12*(2), 34.
- Bravyi, S. et al. (2020). Obstacles to Variational Quantum Optimization from Symmetry Protection. *Phys. Rev. Lett. 125*, 260505.
- Slate, N. et al. (2021). Quantum Walk-Based Vehicle Routing Optimisation. *Quantum, 5*, 513.
- Nielsen, M. A., & Chuang, I. L. *Quantum Computation and Quantum Information*. Chapter 8.

---

## License

Code released under the MIT License. See `LICENSE` for details.
