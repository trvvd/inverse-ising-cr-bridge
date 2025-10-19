# Inverse Ising ↔ Consumer–Resource: an unsupervised stat-mech bridge

**TL;DR**  
A) **Inverse Ising (pseudo-likelihood)** recovers pairwise couplings `J` from samples (node-wise logistic regressions with L1+CV).  
B) **Consumer–Resource (MacArthur)** learns preferences `C` and maintenance costs `m` from time series, then checks feasibility & stability.  
**Bridge)** The effective competition \(A_{\text{eff}} = C D^{-1} C^\top\) **predicts** Ising couplings: ROC-AUC ≈ 0.89 on synthetic data.

## Repro
```bash

make setup
make A     # Part A: writes results/results_core.npz (+ PNGs)
make B     # Part B: reads results_core.npz (n_species), writes results/J_target_from_CR.npz


inverse-ising-cr-bridge/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ Makefile
├─ src/
│  └─ bridge/
│     ├─ inverse_ising.py          # Part A (final script)
│     ├─ consumer_resource.py      # Part B (final script)
│     └─ __init__.py
├─ notebooks/
│  ├─ A_inverse_ising_pl.ipynb     # Part A notebook
│  └─ B_consumer_resource.ipynb    # Part B notebook
├─ results/                        # Part A artifacts & bridge npz/png
└─ cr_results/                     # Part B artifacts & bridge plots
