# Inverse Ising ↔ Consumer–Resource: an unsupervised stat-mech bridge

**TL;DR**  
A) **Inverse Ising (pseudo-likelihood)** recovers pairwise couplings `J` from samples using node-wise logistic regressions with L1+CV.  
B) **Consumer–Resource (MacArthur)** learns preferences `C` and maintenance costs `m` from time series and evaluates feasibility & stability.  
**Bridge)** The effective competition \(A_{\text{eff}} = C D^{-1} C^\top\) **predicts** Ising couplings: we observe ROC-AUC ≈ **0.89** on synthetic data.

<p align="center">
  <img src="docs/images/bridge_scatter.png" width="420" alt="Bridge scatter"/>
</p>

---

## 1. Motivation

Pairwise graphical models are powerful but phenomenological; mechanistic ecological models (MacArthur CR) are interpretable but require dynamics. This repo demonstrates both ends **and** an interpretable **bridge**:
- from dynamics → preferences \((C,m)\) → **effective competition** \(A_{\text{eff}}\),
- which aligns with **pairwise couplings** \(J\) inferred from snapshots.

---

## 2. What’s inside

### Part A — Inverse Ising (pseudo-likelihood)
- Node-wise logistic regressions (`SAGA`, `L1`, `LogisticRegressionCV`) with hold-out conditional log-loss.
- Gibbs sampling to generate synthetic snapshots.
- Recovery metrics: Pearson correlation (upper-tri), ROC-AUC / PR-AUC for edge detection, MSE, sign accuracy, asymmetry of raw coefficients.
- Ablations over sample size, density, and inverse temperature \( \beta \).

### Part B — Consumer–Resource (MacArthur)
- ODE simulation via `solve_ivp` under multiple resource inflow profiles.
- Linearized regression \((\dot n_i/n_i = C_i \cdot R - m_i)\): `LassoCV(positive=True)` → NNLS refinement.
- Feasibility & steady-state checks (residuals); **stability** via Jacobian eigen-spectra.
- Robustness to measurement noise; sample-complexity wrt #experiments and time resolution.

### Bridge — \(A_{\text{eff}}\) ↔ \(J\)
- Build \(A_{\text{eff}}(Ĉ) = Ĉ D^{-1} Ĉ^\top\); compare to \(-J_{\text{ref}}\) (either `J_true` exported from CR or `J_hat` from inverse Ising).
- Report correlation & ROC/PR; estimate scale \( \kappa \) in \( J \approx -\kappa A_{\text{eff}} \).

---

## 3. Quick start

```bash
make setup
make A     # Part A: writes results/results_core.npz (+ PNGs)
make B     # Part B: reads results_core.npz (n_species), writes results/J_target_from_CR.npz




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
│  └─ B_consumer_resource.ipynb    # Part B notebook
├─ results/                        # Part A artifacts & bridge npz/png
└─ cr_results/                     # Part B artifacts & bridge plots
