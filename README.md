# inverse-ising-cr-bridge
Inverse Ising (pseudo-likelihood) — node-wise L1+CV, Gibbs sampling, ROC/PR, conditional log-loss, ablations. Consumer–Resource (MacArthur) — infer Cm from ODE time series (Lasso→NNLS), stability via Jacobian spectra. Bridge: effective competition predicts Ising couplings (ROC-AUC ≈ 0.89).


inverse-ising-cr-bridge/
├─ README.md
├─ LICENSE
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
