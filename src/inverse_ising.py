#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Inverse Ising via Pseudolikelihood — extended version:
- Bridge mode: take J_true from a Consumer–Resource model (if available) BEFORE sampling
- Hold-out validation and conditional log-loss on the test set
- L1/L2 + LogisticRegressionCV (C selection via CV)
- Metrics: corr / ROC-AUC / PR-AUC / MSE, sign accuracy, asymmetry norm
- ROC/PR curves
- (opt.) Ablation disabled by default
- Save results in results/ (npz with meta)
"""

import os
import json
import numpy as np
from numpy.random import default_rng
from scipy.linalg import eigh
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, mean_squared_error,
    roc_curve, precision_recall_curve
)
import sklearn
import scipy

# ------------------------
# Utilities: generate J, fields h, spectral rescale
# ------------------------

def generate_sparse_symmetric_J(N, density=0.1, scale=0.8, rng=None):
    rng = default_rng() if rng is None else rng
    mask_upper = rng.random((N, N))
    mask_upper = np.triu((mask_upper < density).astype(float), k=1)
    vals_upper = rng.normal(loc=0.0, scale=scale, size=(N, N))
    vals_upper = np.triu(vals_upper, k=1)
    J = vals_upper * mask_upper
    J = J + J.T
    np.fill_diagonal(J, 0.0)
    return J

def spectral_rescale(J, target_radius=0.9):
    w = eigh(J, eigvals_only=True)
    r = np.max(np.abs(w))
    if r > 0:
        return (target_radius / r) * J
    return J.copy()

def generate_fields(N, scale_h=0.0, rng=None):
    rng = default_rng() if rng is None else rng
    return rng.normal(loc=0.0, scale=scale_h, size=N)

# ------------------------
# Gibbs sampler
# ------------------------

def gibbs_sampler(J, h, beta=1.0, n_sweeps=5000, burn_in=1000, thin=5, rng=None, init=None):
    rng = default_rng() if rng is None else rng
    N = J.shape[0]
    if init is None:
        s = rng.choice(np.array([-1, 1], dtype=np.int8), size=N)
    else:
        s = np.array(init, dtype=np.int8)
        s[s == 0] = 1
        s = np.sign(s)
        s[s == 0] = 1

    samples = []
    for sweep in range(n_sweeps):
        for i in range(N):
            local_field = h[i] + np.dot(J[i, :], s) - J[i, i] * s[i]
            prob_up = 1.0 / (1.0 + np.exp(-2.0 * beta * local_field))
            s[i] = 1 if rng.random() < prob_up else -1
        if sweep >= burn_in and ((sweep - burn_in) % thin == 0):
            samples.append(s.copy())
    samples = np.array(samples, dtype=np.int8)
    return samples  # shape: (M, N)

# ------------------------
# Pseudolikelihood (node-wise logistic regressions)
# ------------------------

def fit_inverse_ising_pseudolikelihood(
    samples, penalty='l1', use_cv=True, C=1.0, cv_folds=5, Cs=10,
    max_iter=400, n_jobs=-1, intercept=True, random_state=0
):
    """
    For each node i we fit a logistic regression:
    y = 1{s_i=+1}, X = s_{-i}. Coefficients /2 => J_ij, intercept/2 => h_i.
    Returns J_hat (symmetrized), h_hat, raw (before symmetrization).
    """
    M, N = samples.shape
    J_hat_raw = np.zeros((N, N), dtype=float)
    h_hat = np.zeros(N, dtype=float)

    solver = 'saga' if penalty == 'l1' else 'lbfgs'

    if use_cv:
        base_solver = 'saga' if penalty == 'l1' else 'lbfgs'
        def lr_ctor():
            return LogisticRegressionCV(
                penalty=penalty, solver=base_solver,
                cv=cv_folds, Cs=Cs, scoring='neg_log_loss',
                max_iter=max_iter, n_jobs=n_jobs,
                fit_intercept=intercept, random_state=random_state, refit=True
            )
    else:
        def lr_ctor():
            return LogisticRegression(
                penalty=penalty, solver=solver, C=C,
                max_iter=max_iter,
                n_jobs=n_jobs if solver == 'saga' else None,
                fit_intercept=intercept, random_state=random_state
            )

    for i in range(N):
        y = (samples[:, i] == 1).astype(int)
        cols = [j for j in range(N) if j != i]
        X = samples[:, cols].astype(float)
        lr = lr_ctor()
        lr.fit(X, y)
        coefs = lr.coef_.ravel() / 2.0
        row = np.zeros(N, dtype=float)
        row[cols] = coefs
        J_hat_raw[i, :] = row
        h_hat[i] = (lr.intercept_[0] / 2.0) if intercept else 0.0

    J_hat = 0.5 * (J_hat_raw + J_hat_raw.T)
    np.fill_diagonal(J_hat, 0.0)
    return J_hat, h_hat, J_hat_raw

# ------------------------
# Metrics, curves, and helper functions
# ------------------------

def upper_triangular(A):
    N = A.shape[0]
    iu = np.triu_indices(N, k=1)
    return A[iu]

def evaluate_J(J_true, J_pred, eps_zero=1e-12):
    jt = upper_triangular(J_true)
    jp = upper_triangular(J_pred)
    corr = np.corrcoef(jt, jp)[0, 1]
    y_true = (np.abs(jt) > eps_zero).astype(int)
    y_score = np.abs(jp)
    roc = pr = np.nan
    if len(np.unique(y_true)) == 2:
        roc = roc_auc_score(y_true, y_score)
        pr = average_precision_score(y_true, y_score)
    mse = mean_squared_error(J_true.ravel(), J_pred.ravel())
    return {"corr": corr, "roc_auc": roc, "pr_auc": pr, "mse": mse, "y_true": y_true, "y_score": y_score}

def sign_accuracy(J_true, J_pred, eps_zero=1e-12):
    mask = np.abs(J_true) > eps_zero
    if not np.any(mask):
        return np.nan
    return np.mean(np.sign(J_true[mask]) == np.sign(J_pred[mask]))

def asymmetry_fro_norm(J_raw):
    A = J_raw - J_raw.T
    return np.linalg.norm(A, ord='fro')

def conditional_logloss(samples, J, h, beta=1.0):
    """
    - (1 / (M*N)) * sum_{m,i} log P(s_i^{(m)} | s_{-i}^{(m)}),
    P = sigma(2 beta s_i (h_i + sum_j J_ij s_j))
    """
    M, N = samples.shape
    s = samples.astype(float)
    local = s @ J.T + h             # (M, N)
    z = 2.0 * beta * (s * local)    # elementwise
    log_sigma = -np.log1p(np.exp(-z))
    return float(-np.mean(log_sigma))

def plot_heatmaps(J_true, J_pred, savepath=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(J_true, origin='lower', aspect='auto')
    axes[0].set_title("J (true)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(J_pred, origin='lower', aspect='auto')
    axes[1].set_title("J (inferred)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    if savepath: plt.savefig(savepath, dpi=150)
    plt.show()

def plot_scatter(J_true, J_pred, savepath=None, s=12, alpha=0.6):
    jt = upper_triangular(J_true)
    jp = upper_triangular(J_pred)
    lim = max(np.max(np.abs(jt)), np.max(np.abs(jp)))
    lim = max(lim * 1.05, 1e-3)
    plt.figure(figsize=(5, 5))
    plt.scatter(jt, jp, s=s, alpha=alpha)
    plt.plot([-lim, lim], [-lim, lim], 'k--', linewidth=1)
    plt.xlabel("J_true (upper-tri)")
    plt.ylabel("J_pred (upper-tri)")
    plt.title("Scatter: true vs inferred couplings")
    plt.grid(True, ls=':')
    plt.axis('equal'); plt.xlim(-lim, lim); plt.ylim(-lim, lim)
    if savepath: plt.savefig(savepath, dpi=150)
    plt.show()

def plot_roc_pr_curves(y_true, y_score, save_prefix=None):
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)

    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC curve")
    plt.grid(True, ls=':'); 
    if save_prefix: plt.savefig(f"{save_prefix}_roc.png", dpi=150)
    plt.show()

    plt.figure(figsize=(5,4))
    plt.plot(rec, prec); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR curve")
    plt.grid(True, ls=':');
    if save_prefix: plt.savefig(f"{save_prefix}_pr.png", dpi=150)
    plt.show()

# ------------------------
# Ablation (small grid) — disabled by default
# ------------------------

def ablation_study(
    N=40, betas=(0.7, 1.0), densities=(0.07, 0.12), Ms=(1500, 4000),
    J_scale=0.8, target_radius=0.9, seed=123, results_dir="results/ablation"
):
    os.makedirs(results_dir, exist_ok=True)
    rng = default_rng(seed)
    table = []
    for beta in betas:
        for density in densities:
            for M_target in Ms:
                J_true = spectral_rescale(
                    generate_sparse_symmetric_J(N, density=density, scale=J_scale, rng=rng),
                    target_radius=target_radius
                )
                h_true = generate_fields(N, scale_h=0.0, rng=rng)

                thin = 5
                burn_in = 1000
                n_sweeps = burn_in + M_target * thin

                samples = gibbs_sampler(J_true, h_true, beta=beta,
                                        n_sweeps=n_sweeps, burn_in=burn_in, thin=thin, rng=rng)

                M = samples.shape[0]
                idx = rng.permutation(M)
                tr_cut = int(0.8 * M)
                tr_idx, te_idx = idx[:tr_cut], idx[tr_cut:]
                tr, te = samples[tr_idx], samples[te_idx]

                J_hat, h_hat, J_raw = fit_inverse_ising_pseudolikelihood(
                    tr, penalty='l1', use_cv=True, cv_folds=5, Cs=8,
                    max_iter=400, n_jobs=-1, intercept=True, random_state=0
                )

                m = evaluate_J(J_true, J_hat)
                sign_acc = sign_accuracy(J_true, J_hat)
                asymm = asymmetry_fro_norm(J_raw)

                cll_tr = conditional_logloss(tr, J_hat, h_hat, beta=beta)
                cll_te = conditional_logloss(te, J_hat, h_hat, beta=beta)

                row = dict(beta=beta, density=density, M=M, corr=float(m["corr"]),
                           roc_auc=float(m["roc_auc"]) if not np.isnan(m["roc_auc"]) else None,
                           pr_auc=float(m["pr_auc"]) if not np.isnan(m["pr_auc"]) else None,
                           mse=float(m["mse"]), sign_acc=float(sign_acc), asymm=float(asymm),
                           cll_train=float(cll_tr), cll_test=float(cll_te))
                table.append(row)

                with open(os.path.join(results_dir, f"beta{beta}_dens{density}_M{M}.json"), "w") as f:
                    json.dump(row, f, indent=2)
    print("\n== Ablation (summary) ==")
    hdr = ["beta","density","M","corr","roc_auc","pr_auc","mse","sign_acc","asymm","cll_train","cll_test"]
    print("\t".join(hdr))
    for r in table:
        print("\t".join(str(r[k]) for k in hdr))
    return table

# ------------------------
# Main routine
# ------------------------

def main():
    os.makedirs("results", exist_ok=True)

    print("== Library versions ==")
    print(f"numpy: {np.__version__}, scipy: {scipy.__version__}, sklearn: {sklearn.__version__}")

    rng = default_rng(42)

    # --- Sampler parameters ---
    beta = 1.0
    burn_in = 1500
    thin = 5
    total_kept = 6000
    n_sweeps = burn_in + total_kept * thin

    # --- Logistic Regression settings ---
    penalty = 'l1'
    use_cv = True
    C_reg = 0.5           # used if use_cv=False
    cv_folds = 5
    Cs = 8
    max_iter = 400
    n_jobs = -1
    fit_intercept = True

    # === Bridge input from CR (BEFORE sampling) ===
    CR_J_PATH = "results/J_target_from_CR.npz"
    bridge_mode = False
    target_radius = 0.9   # for random J; CR matrix is already rescaled

    if os.path.exists(CR_J_PATH):
        pack = np.load(CR_J_PATH, allow_pickle=True)
        J_true = pack["J_target"].astype(float)
        N = J_true.shape[0]
        h_true = np.zeros(N)  # no external fields
        bridge_mode = True
        print(f"Using J_true from CR (bridge mode), N={N}")
    else:
        # --- fallback: random J_true ---
        N = 50
        density = 0.10
        J_scale = 0.8
        h_scale = 0.0
        J_true = generate_sparse_symmetric_J(N, density=density, scale=J_scale, rng=rng)
        J_true = spectral_rescale(J_true, target_radius=target_radius)
        h_true = generate_fields(N, scale_h=h_scale, rng=rng)
        print(f"Using random J_true, N={N}, density≈{density}")

    # Diagnostics of true J
    w = eigh(J_true, eigvals_only=True)
    rho = np.max(np.abs(w))
    dens_meta = np.count_nonzero(np.triu(np.abs(J_true) > 1e-12, 1)) / (N * (N - 1) / 2)
    print(f"Spectral radius of J_true: {rho:.3f}  |  density(upper)≈{dens_meta:.3f}")

    # == Gibbs sampling ==
    samples = gibbs_sampler(J_true, h_true, beta=beta,
                            n_sweeps=n_sweeps, burn_in=burn_in, thin=thin, rng=rng)
    M = samples.shape[0]
    print(f"Collected samples: M={M} (after burn-in and thinning)")

    # --- Hold-out split ---
    idx = rng.permutation(M)
    tr_cut = int(0.8 * M)
    tr_idx, te_idx = idx[:tr_cut], idx[tr_cut:]
    samples_tr, samples_te = samples[tr_idx], samples[te_idx]
    print(f"Train: {samples_tr.shape[0]}   Test: {samples_te.shape[0]}")

    print("\n== Inference via pseudolikelihood (node-wise LR) ==")
    J_hat, h_hat, J_raw = fit_inverse_ising_pseudolikelihood(
        samples_tr, penalty=penalty, use_cv=use_cv, C=C_reg, cv_folds=cv_folds, Cs=Cs,
        max_iter=max_iter, n_jobs=n_jobs, intercept=fit_intercept, random_state=0
    )

    # --- Metrics for reconstructing J ---
    print("\n== Evaluation of J reconstruction ==")
    metrics = evaluate_J(J_true, J_hat)
    print(f"corr(Pearson, upper-tri): {metrics['corr']:.3f}")
    print(f"ROC-AUC (edge detect):    {metrics['roc_auc']:.3f}" if not np.isnan(metrics['roc_auc']) else "ROC-AUC: n/a")
    print(f"PR-AUC  (edge detect):    {metrics['pr_auc']:.3f}" if not np.isnan(metrics['pr_auc']) else "PR-AUC: n/a")
    print(f"MSE(J):                    {metrics['mse']:.6f}")
    sign_acc = sign_accuracy(J_true, J_hat)
    print(f"Sign accuracy (|J|>0):     {sign_acc:.3f}")
    asymm = asymmetry_fro_norm(J_raw)
    print(f"||J_raw - J_raw^T||_F:     {asymm:.4f}")

    # --- Conditional log-loss on train/test ---
    cll_tr = conditional_logloss(samples_tr, J_hat, h_hat, beta=beta)
    cll_te = conditional_logloss(samples_te, J_hat, h_hat, beta=beta)
    print("\n== Conditional log-loss ==")
    print(f"Train: {cll_tr:.4f}   Test: {cll_te:.4f}")

    # --- Visualizations ---
    print("\n== Visualization ==")
    plot_heatmaps(J_true, J_hat, savepath="results/heatmaps.png")
    plot_scatter(J_true, J_hat, savepath="results/scatter.png")
    plot_roc_pr_curves(metrics["y_true"], metrics["y_score"], save_prefix="results/curves")

    # --- Save artifacts ---
    np.savez_compressed(
        "results/results_core.npz",
        J_true=J_true, J_hat=J_hat, J_raw=J_raw, h_true=h_true, h_hat=h_hat,
        metrics_core=np.array([
            metrics['corr'],
            metrics['roc_auc'] if not np.isnan(metrics['roc_auc']) else -1.0,
            metrics['pr_auc'] if not np.isnan(metrics['pr_auc']) else -1.0,
            metrics['mse'], sign_acc, asymm, cll_tr, cll_te
        ], dtype=float),
        meta=np.array([N, dens_meta, beta, target_radius, M, float(bridge_mode)], dtype=float)
    )
    with open("results/metrics_readable.json","w") as f:
        out = dict(
            corr=float(metrics['corr']),
            roc_auc=None if np.isnan(metrics['roc_auc']) else float(metrics['roc_auc']),
            pr_auc=None if np.isnan(metrics['pr_auc']) else float(metrics['pr_auc']),
            mse=float(metrics['mse']),
            sign_accuracy=float(sign_acc),
            asymmetry_fro_norm=float(asymm),
            conditional_logloss_train=float(cll_tr),
            conditional_logloss_test=float(cll_te),
            N=int(N), density=float(dens_meta), beta=float(beta),
            target_radius=float(target_radius), M=int(M),
            bridge_mode=bool(bridge_mode),
            libs=dict(numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__)
        )
        json.dump(out, f, indent=2, ensure_ascii=False)

    # --- Ablation (disabled by default so it doesn't interfere with bridge) ---
    RUN_ABLATION = False
    if RUN_ABLATION:
        ablation_study(
            N=40, betas=(0.7, 1.0), densities=(0.07, 0.12), Ms=(1500, 3000),
            J_scale=0.8, target_radius=0.9, seed=777, results_dir="results/ablation"
        )

if __name__ == "__main__":
    main()
