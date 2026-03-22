# Quantum-Classical Hybrid TSP: Uttarakhand Tourism Route Optimisation

"Quantum-Classical Hybrid Approaches to the Travelling Salesman Problem:
A Comparative Study on Uttarakhand Tourism Route Optimisation"

## Requirements
Python 3.10+
PennyLane 0.38+
PennyLane-Lightning 0.38+
NumPy 1.24+
NetworkX 3.1+
Pandas 2.0+
Matplotlib 3.7+
Seaborn 0.12+

## Installation
pip install pennylane pennylane-lightning numpy networkx pandas matplotlib seaborn

## Running
python uttarakhand_tsp.py

Output: /figures/ directory with all 12 paper figures and results.json

## Presets
Change PRESET = 'FAST' to 'QUALITY' or 'FINAL' for stronger results.
FAST: 120 steps/layer (used in paper)
QUALITY: 300 steps/layer
FINAL: 500 steps/layer

## Expected runtime
FAST preset: approximately 4.5 minutes on standard CPU
