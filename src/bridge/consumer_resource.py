#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Consumer–Resource (MacArthur): learn preferences and check stability
+ robustness to noise, NNLS refine, sample complexity, F1 threshold,
+ stationarity check (residuals) and a "bridge" to Inverse Ising (A_eff vs J_*).

Auto-sync of dimensions:
- Reads results/results_core.npz from part A and sets n_species = J_hat.shape[0] (if available).
- If the file is missing — uses the default N_SPECIES_DEFAULT.

Dependencies: numpy, scipy, scikit-learn, matplotlib
Run: python consumer_resource_plus.py
"""

import os
import numpy as np
from numpy.random import default_rng
from scipy.integrate import solve_ivp
from scipy.linalg import eigvals
from scipy.optimize import nnls
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LassoCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    mean_squared_error
)

# -------------------------
# Path config and auto-detection of size from Inverse Ising
# -------------------------
ISING_NPZ_PATH = "results/results_core.npz"
N_SPECIES_DEFAULT = 30   # used if NPZ is not found

def detect_n_species_from_ising(path=ISING_NPZ_PATH, default_n=N_SPECIES_DEFAULT):
    p = Path(path)
    if not p.exists():
        print(f"[Init] {path} not found — using n_species={default_n}")
        return default_n
    data = np.load(str(p), allow_pickle=True)
    if "J_hat" not in data:
        print(f"[Init] No 'J_hat' key in {path} — using n_species={default_n}")
        return default_n
    n = int(data["J_hat"].shape[0])
    print(f"[Init] Found Inverse Ising NPZ: n_species={n} (from J_hat)")
    return n

# =========================
# Parameter generation
# =========================

def generate_params(n_species=30, n_res=10, density=0.15,
                    c_scale=1.0, s_scale=2.0, d_scale=1.0, m_scale=0.6, seed=42):
    rng = default_rng(seed)
    mask = rng.random((n_species, n_res)) < density
    C = np.zeros((n_species, n_res))
    C[mask] = rng.gamma(shape=2.0, scale=c_scale/2.0, size=mask.sum())
    s = rng.gamma(shape=2.0, scale=s_scale/2.0, size=n_res) + 0.2
    d = rng.gamma(shape=2.0, scale=d_scale/2.0, size=n_res) + 0.2
    m = rng.gamma(shape=2.0, scale=m_scale/2.0, size=n_species) + 0.05
    return C, s, d, m

# =========================
# CR dynamics
# =========================

def cr_rhs(t, z, C, s, d, m):
    n_res = s.shape[0]; n_species = m.shape[0]
    R = z[:n_res]; n = z[n_res:]
    cons = (C.T @ n)                 # (n_res,)
    dR = s - d*R - R*cons
    growth = (C @ R) - m             # (n_species,)
    dn = n * growth
    return np.concatenate([dR, dn])

def simulate_experiment(C, s, d, m, t_span=(0,60.0), t_eval=None, R0=None, n0=None, seed=0):
    rng = default_rng(seed)
    n_res = s.shape[0]; n_species = m.shape[0]
    if R0 is None: R0 = rng.uniform(0.5, 2.0, size=n_res)
    if n0 is None: n0 = rng.uniform(0.1, 0.5, size=n_species)
    if t_eval is None: t_eval = np.linspace(t_span[0], t_span[1], 601)
    z0 = np.concatenate([R0, n0])
    sol = solve_ivp(cr_rhs, t_span, z0, t_eval=t_eval, args=(C, s, d, m), rtol=1e-6, atol=1e-9)
    R = sol.y[:n_res, :].T
    n = sol.y[n_res:, :].T
    return sol.t, R, n

def integrate_to_steady(C, s, d, m, seed=0, t_end=300.0, tol=1e-8, max_loops=3):
    """Integrate longer until stationarity residual < tol or max_loops is exhausted."""
    rng = default_rng(seed)
    R0 = rng.uniform(0.5, 2.0, size=s.shape[0])
    n0 = rng.uniform(0.1, 0.5, size=m.shape[0])
    t_eval = np.linspace(0, t_end, int(t_end*10)+1)
    for _ in range(max_loops):
        t, R, n = simulate_experiment(C, s, d, m, t_span=(0, t_end), t_eval=t_eval, R0=R0, n0=n0, seed=seed)
        R_star, n_star = R[-1], n[-1]
        res_R, res_n = stationarity_residuals(R_star, n_star, C, s, d, m)
        if max(res_R, res_n) < tol:
            break
        R0, n0 = R_star, n_star
        t_end *= 1.5
        t_eval = np.linspace(0, t_end, int(t_end*10)+1)
    return R_star, n_star

# =========================
# Stationarity residuals
# =========================

def stationarity_residuals(R, n, C, s, d, m):
    cons = C.T @ n
    dR = s - d*R - R*cons
    growth = (C @ R) - m
    dn = n * growth
    res_R = float(np.max(np.abs(dR)))
    res_n = float(np.max(np.abs(dn)))
    return res_R, res_n

# =========================
# Data preparation for regression
# =========================

def time_derivative_central(x, t):
    dx = np.empty_like(x)
    dx[1:-1] = (x[2:] - x[:-2]) / (t[2:] - t[:-2])[:, None]
    dx[0]  = (x[1] - x[0]) / (t[1] - t[0])
    dx[-1] = (x[-1] - x[-2]) / (t[-1] - t[-2])
    return dx

def smooth_time_series(R, n, window=11, poly=2):
    """Light temporal smoothing for a stable derivative."""
    R_sm = savgol_filter(R, window_length=window, polyorder=poly, axis=0, mode='interp')
    n_sm = savgol_filter(n, window_length=window, polyorder=poly, axis=0, mode='interp')
    R_sm = np.clip(R_sm, 0.0, None)
    n_sm = np.clip(n_sm, 0.0, None)
    return R_sm, n_sm

def build_regression_dataset(t_list, R_list, n_list, n_min=1e-4, step=1):
    """Returns X_i, y_i for each species: y_i = dn_i/dt / n_i, X = R."""
    n_species = n_list[0].shape[1]; n_res = R_list[0].shape[1]
    per_species_X = [[] for _ in range(n_species)]
    per_species_y = [[] for _ in range(n_species)]
    for t, R, n in zip(t_list, R_list, n_list):
        t = t[::step]; R = R[::step]; n = n[::step]
        dn = time_derivative_central(n, t)
        y = dn / np.clip(n, n_min, None)
        for i in range(n_species):
            mask = n[:, i] > n_min
            per_species_X[i].append(R[mask])
            per_species_y[i].append(y[mask, i])
    X_per = [np.vstack(xs) if len(xs)>0 else np.zeros((0, n_res)) for xs in per_species_X]
    y_per = [np.concatenate(ys) if len(ys)>0 else np.zeros((0,)) for ys in per_species_y]
    return X_per, y_per

# =========================
# Fitting C, m
# =========================

def fit_lasso_positive(X_per_species, y_per_species, cv=5, random_state=0, max_iter=20000):
    """LassoCV (positive=True, free intercept). Returns (C_hat, m_hat)."""
    n_species = len(X_per_species)
    n_res = X_per_species[0].shape[1]
    C_hat = np.zeros((n_species, n_res)); m_hat = np.zeros(n_species)
    for i in range(n_species):
        X = X_per_species[i]; y = y_per_species[i]
        if X.shape[0] == 0: continue
        model = LassoCV(cv=cv, fit_intercept=True, positive=True,
                        random_state=random_state, n_jobs=-1, max_iter=max_iter)
        model.fit(X, y)
        C_hat[i] = model.coef_
        m_hat[i] = -model.intercept_
    return C_hat, m_hat

def nnls_refine_given_m(X_per_species, y_per_species, m_hat, nonneg_clip=True):
    """NNLS refine: X_i c_i ≈ y_i + m_hat[i],  c_i >= 0."""
    n_species = len(X_per_species); n_res = X_per_species[0].shape[1]
    C_nn = np.zeros((n_species, n_res))
    for i in range(n_species):
        X = X_per_species[i]; y = y_per_species[i]
        if X.shape[0] == 0: continue
        target = y + m_hat[i]
        c_i, _ = nnls(X, target)
        if nonneg_clip: c_i = np.clip(c_i, 0, None)
        C_nn[i] = c_i
    return C_nn

# =========================
# Metrics and thresholds
# =========================

def evaluate_C(C_true, C_hat, eps=1e-12):
    rmse = np.sqrt(mean_squared_error(C_true.ravel(), C_hat.ravel()))
    mae = np.mean(np.abs(C_true - C_hat))
    y_true = (C_true > eps).astype(int).ravel()
    y_score = C_hat.ravel()
    roc = pr = np.nan
    if len(np.unique(y_true)) == 2:
        roc = roc_auc_score(y_true, y_score)
        pr  = average_precision_score(y_true, y_score)
    return dict(rmse=rmse, mae=mae, roc_auc=roc, pr_auc=pr, y_true=y_true, y_score=y_score)

def evaluate_m(m_true, m_hat):
    mae = float(np.mean(np.abs(m_true - m_hat)))
    rmse = float(np.sqrt(np.mean((m_true - m_hat)**2)))
    return dict(mae=mae, rmse=rmse)

def best_f1_threshold(y_true, y_score):
    if len(np.unique(y_true)) < 2: return None, None
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    f1 = 2*prec*rec/(prec+rec+1e-12)
    k = np.nanargmax(f1)
    thr_val = thr[k-1] if k>0 and (k-1) < len(thr) else np.median(y_score)
    return float(f1[k]), float(thr_val)

# =========================
# Stability
# =========================

def jacobian_at_state(R, n, C, s, d, m):
    n_res = len(R); n_species = len(n)
    J_RR = np.zeros((n_res, n_res))
    J_Rn = np.zeros((n_res, n_species))
    J_nR = np.zeros((n_species, n_res))
    J_nn = np.zeros((n_species, n_species))
    cons = C.T @ n
    growth = (C @ R) - m
    for a in range(n_res):
        J_RR[a, a] = -(d[a] + cons[a])
        for i in range(n_species):
            J_Rn[a, i] = -C[i, a] * R[a]
    for i in range(n_species):
        for a in range(n_res):
            J_nR[i, a] = n[i] * C[i, a]
        J_nn[i, i] = growth[i]  # for survivors at steady state ~0
    return np.vstack([np.hstack([J_RR, J_Rn]), np.hstack([J_nR, J_nn])])

def feasibility(n_star, tol=1e-6):
    return float(np.mean(n_star > tol))

# =========================
# Visualization
# =========================

def plot_heatmaps(C_true, C_hat, savepath=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(C_true, origin='lower', aspect='auto')
    axes[0].set_title("C (true)"); plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(C_hat, origin='lower', aspect='auto')
    axes[1].set_title("C (estimated)"); plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    if savepath: plt.savefig(savepath, dpi=150)
    plt.show()

def plot_scatter_C(C_true, C_hat, savepath=None, s=12, alpha=0.6):
    x = C_true.ravel(); y = C_hat.ravel()
    lim = max(np.max(x), np.max(y)); lim = max(lim*1.05, 1e-3)
    plt.figure(figsize=(5,5))
    plt.scatter(x, y, s=s, alpha=alpha)
    plt.plot([0, lim], [0, lim], 'k--', lw=1)
    plt.xlabel("C_true"); plt.ylabel("C_hat"); plt.title("Scatter C")
    plt.grid(True, ls=':'); plt.xlim(0, lim); plt.ylim(0, lim)
    if savepath: plt.savefig(savepath, dpi=150)
    plt.show()

def plot_roc_pr(y_true, y_score, save_prefix=None):
    if len(np.unique(y_true)) < 2: return
    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(5,4)); plt.plot(fpr, tpr); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (edge detect)")
    plt.grid(True, ls=':');
    if save_prefix: plt.savefig(f"{save_prefix}_roc.png", dpi=150)
    plt.show()
    plt.figure(figsize=(5,4)); plt.plot(rec, prec); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR (edge detect)")
    plt.grid(True, ls=':');
    if save_prefix: plt.savefig(f"{save_prefix}_pr.png", dpi=150)
    plt.show()

def plot_sample_complexity(curve_exp, curve_step, save_prefix="cr_results/sample_complexity"):
    plt.figure(figsize=(5,4))
    xs = [k for k,_ in curve_exp]; ys = [m['rmse'] for _,m in curve_exp]
    plt.plot(xs, ys, marker='o'); plt.xlabel("# experiments K"); plt.ylabel("RMSE(C)")
    plt.title("Sample-complexity: by experiments"); plt.grid(True, ls=':')
    plt.savefig(f"{save_prefix}_experiments.png", dpi=150); plt.show()

    plt.figure(figsize=(5,4))
    xs = [st for st,_ in curve_step]; ys = [m['rmse'] for _,m in curve_step]
    plt.plot(xs, ys, marker='o'); plt.xlabel("thinning step"); plt.ylabel("RMSE(C)")
    plt.title("Sample-complexity: by time resolution"); plt.grid(True, ls=':')
    plt.savefig(f"{save_prefix}_time_step.png", dpi=150); plt.show()

# =========================
# Noise / robustness
# =========================

def add_measurement_noise(R_list, n_list, sigma_R=0.02, sigma_n=0.02, seed=0):
    rng = default_rng(seed)
    Rn = []; nn = []
    for R, n in zip(R_list, n_list):
        R_noisy = np.clip(R + rng.normal(0, sigma_R, size=R.shape), 0.0, None)
        n_noisy = np.clip(n + rng.normal(0, sigma_n, size=n.shape), 0.0, None)
        Rn.append(R_noisy); nn.append(n_noisy)
    return Rn, nn

# =========================
# BRIDGE: A_eff(C) vs J_ref (from Inverse Ising NPZ)
# =========================

def bridge_compare_Aeff_vs_J(C_hat, d, path_to_ising_npz=ISING_NPZ_PATH, out_prefix="cr_results/bridge"):
    p = Path(path_to_ising_npz)
    if not p.exists():
        print("[Bridge] Inverse Ising file not found, skipping comparison.")
        return None
    data = np.load(str(p), allow_pickle=True)

    # take J_ref from NPZ: first J_true (if present), otherwise J_hat
    if "J_true" in data.files:
        J_ref, src = data["J_true"].astype(float), "J_true"
    elif "J_hat" in data.files:
        J_ref, src = data["J_hat"].astype(float), "J_hat"
    else:
        print("[Bridge] NPZ has neither J_true nor J_hat — skipping.")
        return None

    n_species = C_hat.shape[0]
    if J_ref.shape[0] != n_species:
        print(f"[Bridge] Shapes mismatch (J_ref {J_ref.shape[0]} vs species {n_species}), skipping.")
        return None

    # A_eff from estimated C
    A_eff = C_hat @ np.diag(1.0/d) @ C_hat.T
    np.fill_diagonal(A_eff, 0.0)

    iu = np.triu_indices(n_species, 1)
    a_raw = A_eff[iu].astype(float)
    j_raw = (-J_ref[iu]).astype(float)       # competition ⇒ J ≈ -κ·A_eff

    # z-score normalization (remove scale/shift)
    a = (a_raw - a_raw.mean()) / (a_raw.std() + 1e-12)
    j = (j_raw - j_raw.mean()) / (j_raw.std() + 1e-12)

    corr = float(np.corrcoef(a, j)[0, 1])

    # binary "truth": is there an edge in J_ref
    y_true = (np.abs(J_ref[iu]) > 1e-12).astype(int)
    score  = a
    roc = roc_auc_score(y_true, score) if len(np.unique(y_true)) == 2 else np.nan
    pr  = average_precision_score(y_true, score) if len(np.unique(y_true)) == 2 else np.nan

    # OLS estimate of κ*: J ≈ -κ·A_eff (without normalization)
    kappa = -np.linalg.lstsq(a_raw.reshape(-1,1), j_raw.reshape(-1,1), rcond=None)[0].ravel()[0]

    print(f"[Bridge] using {src}; corr(A_eff, -{src})={corr:.3f}, ROC-AUC={roc:.3f}, PR-AUC={pr:.3f}, kappa*={kappa:.3f}")

    # plot (in z-score coordinates)
    plt.figure(figsize=(5,5))
    lim = np.max(np.abs(np.concatenate([j, a]))); lim = max(lim*1.05, 1e-3)
    plt.scatter(j, a, s=10, alpha=0.6)
    plt.plot([-lim, lim], [-lim, lim], 'k--', lw=1)
    plt.xlabel(f"-{src} (upper-tri, z-score)")
    plt.ylabel("A_eff (upper-tri, z-score)")
    plt.title("Bridge: A_eff(C_hat) vs -J_ref")
    plt.grid(True, ls=':')
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(f"{out_prefix}_scatter.png", dpi=150); plt.show()

    return dict(corr=corr, roc_auc=float(roc), pr_auc=float(pr), kappa=kappa)

# =========================
# Main routine
# =========================

def main():
    os.makedirs("cr_results", exist_ok=True)
    rng = default_rng(123)

    # ---- auto-detect the number of species from Inverse Ising ----
    n_species = detect_n_species_from_ising(ISING_NPZ_PATH, default_n=N_SPECIES_DEFAULT)
    n_res = 10
    density = 0.15

    # ---- synthetic parameters ----
    C_true, s_base, d_true, m_true = generate_params(
        n_species=n_species, n_res=n_res, density=density,
        c_scale=1.0, s_scale=2.0, d_scale=1.0, m_scale=0.6, seed=11
    )

    # ---- more and more diverse experiments for better learning ----
    n_experiments = 16
    t_list, R_list, n_list = [], [], []
    for k in range(n_experiments):
        # wider spread of resource supplies
        s_k = s_base * rng.uniform(0.6, 1.6, size=n_res)
        t, R, n = simulate_experiment(C_true, s_k, d_true, m_true,
                                      t_span=(0, 60.0), t_eval=np.linspace(0, 60.0, 601),
                                      seed=100 + k)
        t_list.append(t); R_list.append(R); n_list.append(n)

    # (opt.) smoothing before computing derivatives
    DO_SMOOTH = True
    if DO_SMOOTH:
        R_list_s, n_list_s = [], []
        for R, n in zip(R_list, n_list):
            Rs, ns = smooth_time_series(R, n, window=11, poly=2)
            R_list_s.append(Rs); n_list_s.append(ns)
    else:
        R_list_s, n_list_s = R_list, n_list

    # ---- true feasibility/stability + stationarity check ----
    R_star_true, n_star_true = integrate_to_steady(C_true, s_base, d_true, m_true, seed=999, t_end=300.0)
    feas_true = feasibility(n_star_true, tol=1e-6)
    resR, resN = stationarity_residuals(R_star_true, n_star_true, C_true, s_base, d_true, m_true)
    J_true = jacobian_at_state(R_star_true, n_star_true, C_true, s_base, d_true, m_true)
    max_real_true = float(np.max(np.real(eigvals(J_true))))

    # ---- dataset and Lasso ----
    X_per, y_per = build_regression_dataset(t_list, R_list_s, n_list_s, n_min=1e-4, step=1)
    C_hat_lasso, m_hat_lasso = fit_lasso_positive(X_per, y_per, cv=5, random_state=0, max_iter=20000)
    mC = evaluate_C(C_true, C_hat_lasso)
    mm = evaluate_m(m_true, m_hat_lasso)
    f1, thr = best_f1_threshold(mC["y_true"], mC["y_score"])

    print(f"== BASELINE RECONSTRUCTION (Lasso) | n_species={n_species}, n_res={n_res} ==")
    print(f"RMSE(C): {mC['rmse']:.4f}   MAE(C): {mC['mae']:.4f}   ROC-AUC: {mC['roc_auc']:.3f}   PR-AUC: {mC['pr_auc']:.3f}")
    print(f"MAE(m): {mm['mae']:.4f}   RMSE(m): {mm['rmse']:.4f}")
    if f1 is not None: print(f"Best F1={f1:.3f} at threshold |C|>{thr:.4f}")

    # === SANITY: how close A_true is to -J_true from Ising NPZ ===
    A_true = C_true @ np.diag(1.0/d_true) @ C_true.T
    np.fill_diagonal(A_true, 0.0)
    try:
        pack_ising = np.load(ISING_NPZ_PATH, allow_pickle=True)
        if "J_true" in pack_ising.files:
            J_true_from_ising = pack_ising["J_true"]
            iu = np.triu_indices(A_true.shape[0], 1)
            corr_A_vs_minusJ = np.corrcoef(A_true[iu], (-J_true_from_ising[iu]))[0, 1]
            print(f"[Sanity] corr(A_true, -J_true_from_ising) = {corr_A_vs_minusJ:.3f}")
        else:
            print("[Sanity] results_core.npz has no J_true — rerun Inverse Ising in bridge mode.")
    except FileNotFoundError:
        print("[Sanity] results_core.npz not found — run Inverse Ising first.")

    # ---- NNLS refine with fixed \hat m ----
    C_hat_nnls = nnls_refine_given_m(X_per, y_per, m_hat_lasso)
    mC_nn = evaluate_C(C_true, C_hat_nnls)
    print("\n== NNLS refine (fixed m_hat from Lasso) ==")
    print(f"RMSE(C): {mC_nn['rmse']:.4f}   MAE(C): {mC_nn['mae']:.4f}   ROC-AUC: {mC_nn['roc_auc']:.3f}   PR-AUC: {mC_nn['pr_auc']:.3f}")

    # === SANITY: how close A_hat is to A_true (for both Lasso and NNLS)
    iu = np.triu_indices(A_true.shape[0], 1)
    A_hat_lasso = C_hat_lasso @ np.diag(1.0/d_true) @ C_hat_lasso.T
    np.fill_diagonal(A_hat_lasso, 0.0)
    corr_Ahat_Atrue_lasso = np.corrcoef(A_hat_lasso[iu], A_true[iu])[0, 1]
    print(f"[Sanity] corr(A_hat_lasso, A_true) = {corr_Ahat_Atrue_lasso:.3f}")

    A_hat_nnls = C_hat_nnls @ np.diag(1.0/d_true) @ C_hat_nnls.T
    np.fill_diagonal(A_hat_nnls, 0.0)
    corr_Ahat_Atrue_nnls = np.corrcoef(A_hat_nnls[iu], A_true[iu])[0, 1]
    print(f"[Sanity] corr(A_hat_nnls,  A_true) = {corr_Ahat_Atrue_nnls:.3f}")

    # ---- stability for learned parameters ----
    # 1) at the true steady state (R*, n*)
    J_hat_at_true = jacobian_at_state(R_star_true, n_star_true, C_hat_lasso, s_base, d_true, m_hat_lasso)
    max_real_hat_at_true = float(np.max(np.real(eigvals(J_hat_at_true))))
    # 2) steady state of the system with learned parameters
    R_star_hat, n_star_hat = integrate_to_steady(C_hat_lasso, s_base, d_true, m_hat_lasso, seed=777, t_end=300.0)
    feas_hat = feasibility(n_star_hat, tol=1e-6)
    resR_hat, resN_hat = stationarity_residuals(R_star_hat, n_star_hat, C_hat_lasso, s_base, d_true, m_hat_lasso)
    J_hat = jacobian_at_state(R_star_hat, n_star_hat, C_hat_lasso, s_base, d_true, m_hat_lasso)
    max_real_hat = float(np.max(np.real(eigvals(J_hat))))

    print("\n== STABILITY ==")
    print(f"Feasibility: true {feas_true:.2f} | estimated {feas_hat:.2f}")
    print(f"Stationarity residuals (true): dR {resR:.2e}, dn {resN:.2e}")
    print(f"Stationarity residuals (estimated): dR {resR_hat:.2e}, dn {resN_hat:.2e}")
    print(f"max Re(λ) at true steady state: true {max_real_true:.4f} | estimated {max_real_hat_at_true:.4f}")
    print(f"max Re(λ) at estimated steady state: {max_real_hat:.4f}")

    # ---- visualizations ----
    os.makedirs("cr_results", exist_ok=True)
    plot_heatmaps(C_true, C_hat_lasso, savepath="cr_results/C_heatmaps_lasso.png")
    plot_scatter_C(C_true, C_hat_lasso, savepath="cr_results/C_scatter_lasso.png")
    plot_roc_pr(mC["y_true"], mC["y_score"], save_prefix="cr_results/C_curves_lasso")

    plot_heatmaps(C_true, C_hat_nnls, savepath="cr_results/C_heatmaps_nnls.png")
    plot_scatter_C(C_true, C_hat_nnls, savepath="cr_results/C_scatter_nnls.png")
    plot_roc_pr(mC_nn["y_true"], mC_nn["y_score"], save_prefix="cr_results/C_curves_nnls")

    # ---- robustness to noise ----
    DO_NOISE = True
    if DO_NOISE:
        R_noisy, n_noisy = add_measurement_noise(R_list, n_list, sigma_R=0.03, sigma_n=0.03, seed=321)
        Xn, yn = build_regression_dataset(t_list, R_noisy, n_noisy, n_min=1e-4, step=1)
        C_hat_noise, m_hat_noise = fit_lasso_positive(Xn, yn, cv=5, random_state=0, max_iter=20000)
        mC_noise = evaluate_C(C_true, C_hat_noise); mm_noise = evaluate_m(m_true, m_hat_noise)
        print("\n== Robustness to noise (σ_R=σ_n=0.03) ==")
        print(f"RMSE(C): {mC_noise['rmse']:.4f}   PR-AUC: {mC_noise['pr_auc']:.3f}   MAE(m): {mm_noise['mae']:.4f}")

    # ---- sample-complexity ----
    DO_SAMPLE_COMPLEXITY = True
    if DO_SAMPLE_COMPLEXITY:
        curve_exp = []
        for K in range(1, n_experiments+1):
            Xk, yk = build_regression_dataset(t_list[:K], R_list_s[:K], n_list_s[:K], n_min=1e-4, step=1)
            Ck, mk = fit_lasso_positive(Xk, yk, cv=5, random_state=0, max_iter=20000)
            mkC = evaluate_C(C_true, Ck)
            curve_exp.append((K, mkC))
        curve_step = []
        for step in [1,2,4,8]:
            Xs, ys = build_regression_dataset(t_list, R_list_s, n_list_s, n_min=1e-4, step=step)
            Cs, ms = fit_lasso_positive(Xs, ys, cv=5, random_state=0, max_iter=20000)
            msC = evaluate_C(C_true, Cs)
            curve_step.append((step, msC))
        plot_sample_complexity(curve_exp, curve_step, save_prefix="cr_results/sample_complexity")

    # === export target for Ising from CR ===
    A = C_true @ np.diag(1.0/d_true) @ C_true.T
    np.fill_diagonal(A, 0.0)

    # competition -> J_target ~ -A (scale unimportant; rescale by spectral radius)
    def spectral_rescale(M, target_radius=0.9):
        w = np.linalg.eigvalsh(M)
        r = np.max(np.abs(w))
        return (target_radius/r) * M if r > 0 else M.copy()

    J_target = spectral_rescale(-A, target_radius=0.9)
    os.makedirs("results", exist_ok=True)
    np.savez("results/J_target_from_CR.npz", J_target=J_target)
    print("Saved bridge target -> results/J_target_from_CR.npz, shape:", J_target.shape)

    # === SANITY: if Inverse Ising already ran, A_true vs -J_true_from_ising
    try:
        pack_ising = np.load(ISING_NPZ_PATH, allow_pickle=True)
        if "J_true" in pack_ising.files:
            J_true_from_ising = pack_ising["J_true"]
            iu = np.triu_indices(A.shape[0], 1)
            corr_A_vs_minusJ = np.corrcoef(A[iu], (-J_true_from_ising[iu]))[0, 1]
            print(f"[Sanity] corr(A_true(final), -J_true_from_ising) = {corr_A_vs_minusJ:.3f}")
    except FileNotFoundError:
        pass

    # ---- BRIDGE to Inverse Ising: use NNLS estimate (better for A_eff) ----
    bridge_compare_Aeff_vs_J(C_hat_nnls, d_true, path_to_ising_npz=ISING_NPZ_PATH,
                             out_prefix="cr_results/bridge_Aeff_vs_J")

    # (opt.) Oracle upper bound — what if we had perfect C_true
    # bridge_compare_Aeff_vs_J(C_true, d_true, path_to_ising_npz=ISING_NPZ_PATH,
    #                          out_prefix="cr_results/bridge_oracle")

    # ---- save artifacts ----
    np.savez("cr_results/results_all.npz",
        C_true=C_true, m_true=m_true, s_base=s_base, d_true=d_true,
        C_hat_lasso=C_hat_lasso, m_hat_lasso=m_hat_lasso,
        C_hat_nnls=C_hat_nnls,
        R_star_true=R_star_true, n_star_true=n_star_true,
        R_star_hat=R_star_hat, n_star_hat=n_star_hat,
        feas_true=feas_true, feas_hat=feas_hat,
        max_real_true=max_real_true, max_real_hat_at_true=max_real_hat_at_true, max_real_hat=max_real_hat
    )

if __name__ == "__main__":
    main()
