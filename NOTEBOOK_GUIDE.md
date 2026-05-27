# Notebook Guide
## `hybrid_quantum_classical_tsp_22_cities.ipynb`

This guide explains the structure of the notebook so reviewers and readers know exactly which cells correspond to the paper's results, and which are exploratory.

---

## Two Phases at a Glance

| Phase | Cells | Preset | Purpose | Paper Source? |
|---|---|---|---|---|
| Exploratory | 1 – 6 | FAST | Development & sanity check | ❌ No |
| Paper results | **7 onward** | QUALITY | All tables, all figures | ✅ Yes |

---

## Phase 1 — Cells 1–6 (FAST Preset, Exploratory)

These cells were used during algorithm development to quickly check that the pipeline ran end-to-end. They use a reduced preset (`FAST`: ensemble_size=80, steps_per_layer=120, p_layers=3) and take approximately 3–5 minutes to run.

**None of these outputs appear in the paper.**

| Cell | What it does |
|---|---|
| Cell 1 | Imports, city coordinates, distance matrix, all algorithm definitions (Greedy, 2-Opt, 3-Opt, SA, QAOA, Hybrid, AQP), FAST preset run, saves `results.json` |
| Cells 2–6 | Intermediate figure drafts, route visualisations, VQE variant, preliminary convergence checks — all exploratory |

---

## Phase 2 — Cells 7 onward (QUALITY Preset — Paper Source)

Starting from Cell 7, the notebook switches to the QUALITY preset (`ensemble_size=100, steps_per_layer=300, p_layers=3`) and runs all experiments that generated the paper's tables and figures.

**All numbers in the paper come from this phase.**

| Cell range | What it does | 
|---|---|
| Cell 7 | Full QUALITY preset run — Greedy, 2-Opt, 3-Opt, SA, QAOA, Hybrid, AQP on all 22 cities | 
| Subsequent cells | Recreating All Figures| 
| Subsequent cells | Available keys in results.json | 
| Subsequent cells | Running QAOA Circuit Depth p=5 on 8-qubit subset | 
| Subsequent cells | [Fig 10] Recreated with p=5 included | 
| Subsequent cells | Improvement Analysis | 
| Subsequent cells | Re-running QAOA p=3 with Multiple Random Seeds | 
| Subsequent cells | COMPLETE QAOA CIRCUIT DEPTH ANALYSIS (p=1 to 5) | 
| Subsequent cells | Correcting Figures | 

---

## How to Reproduce Paper Results

### Fastest: Use Pre-Computed JSON

The outputs from the full QUALITY run are already saved in the repo:

- `results.json` — all main algorithm results (22-city, QUALITY preset)
- `qaoa_p5_results.json` — verified p=5 QAOA run on 8-city quantum subset

Run only the figure-regeneration cells (at the end of the notebook) to regenerate all plots from these JSON files. This takes under 1 minute.

### Full Reproduction from Scratch

1. Open the notebook
2. **Skip Cells 1–6** (these are FAST preset exploratory runs)
3. **Start execution from Cell 7**
4. Update `OUTPUT_DIR` in Cell 7 to a directory on your machine
5. Allow approximately **8–10 hours** on a standard CPU

> Expected output from Cell 7: AQP-QAG final tour = 882.95 km, time ≈ 460,732 ms

---

## Notes on the `.py` File

The file `hybrid_quantum_classical_tsp_22_cities.py` is a direct export of the notebook. It contains all cells in sequence, with `# In[N]:` markers indicating the original cell numbers.

**Important:** The script is set to `PRESET = 'FAST'` at line 81. Change this to `PRESET = 'QUALITY'` and update `CUSTOM_OUTPUT_DIR` before running it as a standalone script.

---

## Seed Sensitivity Note (Table 12, p=3)

The p=3 row in Table 12 reports **523.76 km** — this is the best result across 5 random seeds (seed=456 on the 8-city quantum subset). With seed=42, p=3 produces 823.78 km. This seed sensitivity is a known property of QAOA on near-term hardware and is discussed in Section 6.5 of the paper.

---

## Output Directory Note

The notebook and `.py` file contain hardcoded Windows paths (`C:\Users\Aditya Singh\...`) from the original development environment. Update these to your local path before running. All output is saved to that directory: JSON files and PNG figures.
