import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple


# --------------------------
# Helpers
# --------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def topk_neighbors_by_quality(q_row: np.ndarray, k: int, min_quality: float) -> np.ndarray:
    idx = np.where(q_row >= min_quality)[0]
    if idx.size == 0:
        return np.array([], dtype=int)
    idx_sorted = idx[np.argsort(q_row[idx])[::-1]]
    return idx_sorted[:k]


def distance_from_rssi(rssi: np.ndarray, r0: float = -40.0, eta_inv: float = 2.0, d0: float = 1.0) -> np.ndarray:
    return d0 * (10.0 ** ((r0 - rssi) / (10.0 * eta_inv)))


# --------------------------
# (1) RSSI with distance-dependent GMM outliers
# --------------------------

def rssi_from_distance_nlos_gmm(
    d: np.ndarray,
    r0: float = -40.0,
    eta_true: float = 2.0,
    d0: float = 1.0,
    # outlier probability ramp
    p_outlier_min: float = 0.02,
    p_outlier_max: float = 0.20,
    d_outlier_knee: float = 20.0,
    d_outlier_steep: float = 10.0,
    # LOS noise
    sigma_los: float = 2.0,
    # outlier mixture: some too-strong, some too-weak
    p_pos_outlier: float = 0.50,
    mu_outlier_pos: float = 6.0,
    mu_outlier_neg: float = -8.0,
    sigma_outlier: float = 6.0,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      rssi          : RSSI matrix
      is_outlier    : bool mask of outlier links
      outlier_is_pos: bool mask for positive outliers (only meaningful where is_outlier is True)
    """
    if rng is None:
        rng = np.random.default_rng()

    d_safe = np.maximum(d, 1e-3)
    mean_rssi = r0 - 10.0 * eta_true * np.log10(d_safe / d0)

    # distance-dependent p_out
    p_out = p_outlier_min + (p_outlier_max - p_outlier_min) * sigmoid((d_safe - d_outlier_knee) / d_outlier_steep)

    is_out = rng.random(size=d_safe.shape) < p_out
    out_is_pos = np.zeros_like(is_out, dtype=bool)
    out_is_pos[is_out] = rng.random(size=np.count_nonzero(is_out)) < p_pos_outlier

    noise = np.empty(d_safe.shape, dtype=float)

    # LOS
    n_los = np.count_nonzero(~is_out)
    if n_los > 0:
        noise[~is_out] = rng.normal(0.0, sigma_los, size=n_los)

    # Outliers
    n_out = np.count_nonzero(is_out)
    if n_out > 0:
        out_noise = np.empty(n_out, dtype=float)
        pos_mask_flat = out_is_pos[is_out]
        n_pos = np.count_nonzero(pos_mask_flat)
        n_neg = n_out - n_pos
        if n_pos > 0:
            out_noise[pos_mask_flat] = rng.normal(mu_outlier_pos, sigma_outlier, size=n_pos)
        if n_neg > 0:
            out_noise[~pos_mask_flat] = rng.normal(mu_outlier_neg, sigma_outlier, size=n_neg)
        noise[is_out] = out_noise

    return mean_rssi + noise, is_out, out_is_pos


# --------------------------
# (2) Correlated GNSS bias
# --------------------------

def sample_spatially_correlated_bias(
    p_true: np.ndarray,
    sigma_bias: float = 6.0,
    corr_len: float = 20.0,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Draw per-node bias b_i in R^2 with spatial correlation:
      Cov(bx_i, bx_j) = sigma_bias^2 * exp(-d_ij^2/(2*corr_len^2))
    Same for y. x/y fields independent.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = p_true.shape[0]
    d = np.linalg.norm(p_true[:, None, :] - p_true[None, :, :], axis=2)
    K = (sigma_bias ** 2) * np.exp(-(d ** 2) / (2.0 * corr_len ** 2))
    K = K + 1e-6 * np.eye(n)

    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(K + 1e-3 * np.eye(n))

    bx = L @ rng.normal(0.0, 1.0, size=n)
    by = L @ rng.normal(0.0, 1.0, size=n)
    return np.stack([bx, by], axis=1)


# --------------------------
# Gauss-Newton refinement (returns iterations used for cost)
# --------------------------

def refine_with_ranges_gauss_newton(
    p0: np.ndarray,
    Sigma_prior: np.ndarray,
    neigh_pos: np.ndarray,
    d_hat: np.ndarray,
    sigma_d: np.ndarray,
    omega: np.ndarray,
    iters: int = 6
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    min_p (p - p0)^T Sigma_prior^-1 (p - p0) +
         sum_j omega_j * (||p - p_j|| - d_hat_j)^2 / sigma_d_j^2
    Returns (p_hat, Sigma_hat, iters_used).
    """
    W_prior = np.linalg.inv(Sigma_prior)
    p = p0.copy()
    it_used = 0

    for k in range(iters):
        it_used = k + 1
        g = 2.0 * (W_prior @ (p - p0))
        H = 2.0 * W_prior

        for pj, dj, sdj, wj in zip(neigh_pos, d_hat, sigma_d, omega):
            if wj <= 0:
                continue
            v = p - pj
            r = np.linalg.norm(v)
            if r < 1e-6:
                continue
            res = (r - dj)
            J = (v / r).reshape(1, 2)
            w = wj / (sdj ** 2)
            g += 2.0 * w * (J.T.flatten() * res)
            H += 2.0 * w * (J.T @ J)

        try:
            delta = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break

        p = p + delta
        if np.linalg.norm(delta) < 1e-4:
            break

    H_reg = H + 1e-8 * np.eye(2)
    Sigma_hat = np.linalg.inv(H_reg)
    return p, Sigma_hat, it_used


# --------------------------
# Simulation: returns mean error per epoch + mean cost per epoch (per node)
# --------------------------

def simulate_stats_by_epoch(
    n_nodes: int,
    epochs: int,
    neighbor_budget: int,
    grid_size: float,
    seed: int,
    # link-quality mapping
    r_min: float,
    kappa: float,
    min_quality: float,
    # GN
    gn_iters: int,
    # RSSI parameters
    r0: float,
    eta_true: float,
    eta_inv: float,
    p_outlier_min: float,
    p_outlier_max: float,
    d_outlier_knee: float,
    d_outlier_steep: float,
    sigma_los: float,
    p_pos_outlier: float,
    mu_outlier_pos: float,
    mu_outlier_neg: float,
    sigma_outlier: float,
    # correlated bias params
    sigma_bias: float,
    corr_len: float,
    bias_drift_std: float,
    # sigma_d model
    sigma_d_base: float,
    sigma_d_scale: float,
    sigma_d_log_std: float,
    p_confident_wrong: float,
    confident_wrong_scale: float,
    # safety rails + trust (RQ3-style)
    d_min: float,
    d_max: float,
    sigma_d_floor: float,
    lambda_trust: float,
    eps_gate: float,
    w_cap: float
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    # True positions
    p_true = rng.uniform(0.0, grid_size, size=(n_nodes, 2))

    # Per-node uncorrelated GNSS noise covariance (heterogeneous)
    sx = rng.uniform(3.0, 12.0, size=n_nodes)
    sy = rng.uniform(3.0, 12.0, size=n_nodes)
    Sigma_uncorr = np.array([np.diag([sx_i**2, sy_i**2]) for sx_i, sy_i in zip(sx, sy)])

    # correlated GNSS bias field
    bias = sample_spatially_correlated_bias(p_true, sigma_bias=sigma_bias, corr_len=corr_len, rng=rng)
    bias_t = bias.copy()

    # Precompute distances (static)
    d_true = np.linalg.norm(p_true[:, None, :] - p_true[None, :, :], axis=2)

    # Storage
    p_raw = np.zeros((epochs, n_nodes, 2))
    p_hat = np.zeros((epochs, n_nodes, 2))

    # Raw fixes per epoch: true + correlated bias + independent noise (+ drift)
    for t in range(epochs):
        if t > 0 and bias_drift_std > 0:
            bias_t = bias_t + rng.normal(0.0, bias_drift_std, size=bias_t.shape)

        noise_uncorr = np.array([rng.multivariate_normal([0.0, 0.0], Sigma_uncorr[i]) for i in range(n_nodes)])
        p_raw[t] = p_true + bias_t + noise_uncorr

    # Epoch 0
    p_hat[0] = p_raw[0]

    # Track cost per epoch (sum over nodes, later normalized)
    cost_per_epoch = np.zeros(epochs, dtype=float)
    neigh_used_per_epoch = np.zeros(epochs, dtype=float)

    # Iterate
    for t in range(1, epochs):
        # RSSI with distance-dependent outliers
        rssi, is_out, out_is_pos = rssi_from_distance_nlos_gmm(
            d_true,
            r0=r0,
            eta_true=eta_true,
            p_outlier_min=p_outlier_min,
            p_outlier_max=p_outlier_max,
            d_outlier_knee=d_outlier_knee,
            d_outlier_steep=d_outlier_steep,
            sigma_los=sigma_los,
            p_pos_outlier=p_pos_outlier,
            mu_outlier_pos=mu_outlier_pos,
            mu_outlier_neg=mu_outlier_neg,
            sigma_outlier=sigma_outlier,
            rng=rng
        )

        # Link quality from RSSI
        q = sigmoid((rssi - r_min) / kappa)
        np.fill_diagonal(q, 0.0)

        send_pos = p_hat[t - 1]

        total_cost_t = 0.0
        total_neigh_t = 0.0

        for i in range(n_nodes):
            neigh_idx = topk_neighbors_by_quality(q[i], neighbor_budget, min_quality=min_quality)
            k_neigh = int(neigh_idx.size)

            if k_neigh == 0:
                p_hat[t, i] = p_raw[t, i]
                continue

            rssi_ij = rssi[i, neigh_idx]
            q_ij = q[i, neigh_idx]

            # RSSI -> range
            d_hat = distance_from_rssi(rssi_ij, r0=r0, eta_inv=eta_inv)
            d_hat = np.clip(d_hat, d_min, d_max)

            # sigma_d model (miscalibration/non-monotonic)
            sigma_d = sigma_d_base + sigma_d_scale * (1.0 - q_ij)
            sigma_d *= np.exp(rng.normal(0.0, sigma_d_log_std, size=sigma_d.shape))

            out_pos_ij = is_out[i, neigh_idx] & out_is_pos[i, neigh_idx]
            if np.any(out_pos_ij):
                idxs = np.where(out_pos_ij)[0]
                flip = rng.random(size=idxs.size) < p_confident_wrong
                sigma_d[idxs[flip]] *= confident_wrong_scale

            sigma_d = np.maximum(sigma_d, sigma_d_floor)

            # Trust (normalized residual)
            pred_dist = np.linalg.norm(p_raw[t, i] - send_pos[neigh_idx], axis=1)
            eps = np.abs(pred_dist - d_hat) / sigma_d
            s = np.exp(-(eps**2) / (2.0 * lambda_trust**2))
            if eps_gate is not None and eps_gate > 0:
                s[eps > eps_gate] = 0.0

            omega = q_ij * s
            omega = np.minimum(omega, w_cap * (sigma_d**2))

            p_i_hat, _, it_used = refine_with_ranges_gauss_newton(
                p0=p_raw[t, i],
                Sigma_prior=Sigma_uncorr[i],
                neigh_pos=send_pos[neigh_idx],
                d_hat=d_hat,
                sigma_d=sigma_d,
                omega=omega,
                iters=gn_iters
            )
            p_hat[t, i] = p_i_hat

            # --- COST UNITS ---
            # neighbor preprocessing ~ k_neigh
            # GN constraint processing ~ it_used * k_neigh
            total_cost_t += k_neigh + it_used * k_neigh
            total_neigh_t += k_neigh

        # Normalize per node
        cost_per_epoch[t] = total_cost_t / max(n_nodes, 1)
        neigh_used_per_epoch[t] = total_neigh_t / max(n_nodes, 1)

    # Errors
    err = np.linalg.norm(p_hat - p_true[None, :, :], axis=2)   # (epochs, n_nodes)
    mean_err = err.mean(axis=1)                                # (epochs,)

    return {
        "mean_err": mean_err,
        "cost_per_node": cost_per_epoch,           # units per node per epoch
        "neigh_used_per_node": neigh_used_per_epoch
    }


# --------------------------
# Scaling study: budgets 1..max_budget
# --------------------------

def run_scaling_neighbors_vs_error_and_cost(
    n_nodes: int,
    epochs: int,
    grid_size: float,
    runs: int,
    base_seed: int,
    max_budget: Optional[int],
    sim_params: Dict[str, float]
) -> Dict[str, np.ndarray]:
    if max_budget is None:
        max_budget = n_nodes

    budgets = np.arange(1, max_budget + 1)
    B = budgets.size

    # store per budget, per run
    final_err = np.zeros((B, runs), dtype=float)
    final_cost = np.zeros((B, runs), dtype=float)
    mean_cost_over_time = np.zeros((B, runs), dtype=float)

    for bi, b in enumerate(budgets):
        for r in range(runs):
            seed = base_seed + 10000 * int(b) + r
            stats = simulate_stats_by_epoch(
                n_nodes=n_nodes,
                epochs=epochs,
                neighbor_budget=int(b),
                grid_size=grid_size,
                seed=seed,
                **sim_params
            )
            final_err[bi, r] = stats["mean_err"][-1]
            final_cost[bi, r] = stats["cost_per_node"][-1]
            mean_cost_over_time[bi, r] = stats["cost_per_node"][1:].mean()  # average over epochs (exclude t=0)

        print(
            f"budget={int(b):2d}/{max_budget} | "
            f"final err={final_err[bi].mean():.3f} m | "
            f"avg cost/node/epoch={mean_cost_over_time[bi].mean():.1f} units"
        )

    return {
        "budgets": budgets,
        "final_err_mean": final_err.mean(axis=1),
        "final_err_std": final_err.std(axis=1, ddof=1),
        "cost_mean_over_time": mean_cost_over_time.mean(axis=1),
        "cost_std_over_time": mean_cost_over_time.std(axis=1, ddof=1)
    }


def plot_rq4_error_and_cost(res: Dict[str, np.ndarray]) -> None:
    b = res["budgets"]

    # 1) budget vs final error
    plt.figure()
    plt.errorbar(b, res["final_err_mean"], yerr=res["final_err_std"], capsize=3)
    plt.xlabel("Neighbor budget $n$")
    plt.ylabel("Final mean localization error (m)")
    plt.title("RQ4: Neighbor budget vs final error (mean ± std over runs)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 2) budget vs cost
    plt.figure()
    plt.errorbar(b, res["cost_mean_over_time"], yerr=res["cost_std_over_time"], capsize=3)
    plt.xlabel("Neighbor budget $n$")
    plt.ylabel("Compute cost per node per epoch (work units)")
    plt.title("RQ4: Neighbor budget vs compute cost (mean ± std over runs)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 3) Pareto: error vs cost
    plt.figure()
    plt.plot(res["cost_mean_over_time"], res["final_err_mean"], marker="o", linestyle="-")
    for i, n in enumerate(b):
        if n in [1, 5, 10, 15, 20, 25, 30, 40, 50]:
            plt.annotate(str(int(n)), (res["cost_mean_over_time"][i], res["final_err_mean"][i]))
    plt.xlabel("Compute cost per node per epoch (work units)")
    plt.ylabel("Final mean localization error (m)")
    plt.title("RQ4: Accuracy–cost tradeoff (Pareto-style)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    # =========================================================
    # MAIN EXPERIMENT SETTINGS
    # =========================================================
    N_NODES = 50
    EPOCHS = 30
    GRID_SIZE = 100.0
    RUNS = 100
    BASE_SEED = 1
    MAX_BUDGET = N_NODES

    # =========================================================
    # PARAMETERS (same as your realism + safety rails)
    # =========================================================
    sim_params = {
        # link quality mapping
        "r_min": -85.0,
        "kappa": 5.0,
        "min_quality": 0.10,

        # GN
        "gn_iters": 6,

        # RSSI model + inversion mismatch
        "r0": -40.0,
        "eta_true": 2.0,
        "eta_inv": 2.0,

        # (1) distance-dependent outliers
        "p_outlier_min": 0.02,
        "p_outlier_max": 0.20,
        "d_outlier_knee": 20.0,
        "d_outlier_steep": 10.0,
        "sigma_los": 2.0,
        "p_pos_outlier": 0.50,
        "mu_outlier_pos": 6.0,
        "mu_outlier_neg": -8.0,
        "sigma_outlier": 6.0,

        # (2) correlated GNSS bias
        "sigma_bias": 6.0,
        "corr_len": 20.0,
        "bias_drift_std": 0.0,

        # (4) sigma_d miscalibration / non-monotone
        "sigma_d_base": 2.0,
        "sigma_d_scale": 10.0,
        "sigma_d_log_std": 0.25,
        "p_confident_wrong": 0.10,
        "confident_wrong_scale": 0.70,

        # safety rails + trust
        "d_min": 1.0,
        "d_max": 80.0,
        "sigma_d_floor": 5.0,
        "lambda_trust": 2.0,
        "eps_gate": 3.0,
        "w_cap": 0.2
    }

    res = run_scaling_neighbors_vs_error_and_cost(
        n_nodes=N_NODES,
        epochs=EPOCHS,
        grid_size=GRID_SIZE,
        runs=RUNS,
        base_seed=BASE_SEED,
        max_budget=MAX_BUDGET,
        sim_params=sim_params
    )

    plot_rq4_error_and_cost(res)
