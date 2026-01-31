import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple

# ============================================================
# Helpers
# ============================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def rssi_from_distance(
    d: np.ndarray,
    r0: float = -40.0,
    eta: float = 2.0,
    d0: float = 1.0,
    noise_std: float = 2.0,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Log-distance path-loss model for RSSI."""
    if rng is None:
        rng = np.random.default_rng()
    d_safe = np.maximum(d, 1e-3)
    return r0 - 10.0 * eta * np.log10(d_safe / d0) + rng.normal(0.0, noise_std, size=d_safe.shape)


def distance_from_rssi(rssi: np.ndarray, r0: float = -40.0, eta: float = 2.0, d0: float = 1.0) -> np.ndarray:
    """Invert log-distance model: RSSI -> distance."""
    return d0 * (10.0 ** ((r0 - rssi) / (10.0 * eta)))


def topk_neighbors_by_quality(q_row: np.ndarray, k: int, min_quality: float) -> np.ndarray:
    """Pick up to k neighbors with the highest link quality."""
    candidates = np.where(q_row >= min_quality)[0]
    if candidates.size == 0:
        return np.array([], dtype=int)
    order = np.argsort(q_row[candidates])[::-1]
    return candidates[order][:k]


# ============================================================
# Gauss-Newton refinement using range constraints
# ============================================================

def refine_with_ranges_gauss_newton(
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
    neigh_pos: np.ndarray,
    d_hat: np.ndarray,
    sigma_d: np.ndarray,
    omega: np.ndarray,
    iters: int = 6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Minimize:
      (p - prior_mean)^T prior_cov^-1 (p - prior_mean) +
      sum_j omega_j * (||p - p_j|| - d_hat_j)^2 / sigma_d_j^2

    Returns refined p and approximate covariance (~ inverse Hessian).
    """
    W_prior = np.linalg.inv(prior_cov)
    p = prior_mean.copy()

    for _ in range(iters):
        g = 2.0 * (W_prior @ (p - prior_mean))
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
    return p, Sigma_hat


# ============================================================
# Simulation with one cold-start device (no local fix for K epochs)
# ============================================================

def simulate_with_one_cold_start_device(
    n_nodes: int = 50,
    epochs: int = 30,
    neighbor_budget: int = 4,
    grid_size: float = 100.0,
    seed: int = 33,
    cold_start_device: int = 0,
    cold_start_epochs: int = 10,
    gamma_boot: float = 1000.0,  # weak prior inflation (>> 1)
    # Link-quality params
    r0: float = -40.0,
    eta: float = 2.0,
    rssi_noise_std: float = 2.0,
    r_min: float = -85.0,
    kappa: float = 5.0,
    min_quality: float = 0.10,
    # Range uncertainty params
    sigma_d_base: float = 2.0,
    sigma_d_scale: float = 10.0,
    # Optional soft gating
    use_trust: bool = True,
    tau_consistency: float = 25.0,
    gn_iters: int = 6
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    # True positions
    p_true = rng.uniform(0.0, grid_size, size=(n_nodes, 2))

    # Heterogeneous raw covariance per node
    sigma_x = rng.uniform(3.0, 12.0, size=n_nodes)
    sigma_y = rng.uniform(3.0, 12.0, size=n_nodes)
    Sigma_raw = np.array([np.diag([sx**2, sy**2]) for sx, sy in zip(sigma_x, sigma_y)])

    # Raw fixes and refined estimates
    p_raw = np.zeros((epochs, n_nodes, 2))
    p_hat = np.zeros((epochs, n_nodes, 2))
    Sigma_hat = np.zeros((epochs, n_nodes, 2, 2))

    # Generate raw fixes (we will "ignore" them for the cold-start device early on)
    for t in range(epochs):
        for i in range(n_nodes):
            p_raw[t, i] = p_true[i] + rng.multivariate_normal([0.0, 0.0], Sigma_raw[i])

    # Epoch 0 initialization: refined = raw (except cold-start device)
    p_hat[0] = p_raw[0]
    for i in range(n_nodes):
        Sigma_hat[0, i] = Sigma_raw[i]

    # Cold-start device: no local fix for first K epochs
    center = np.array([grid_size / 2.0, grid_size / 2.0])
    p_hat[0, cold_start_device] = center
    Sigma_hat[0, cold_start_device] = gamma_boot * Sigma_raw[cold_start_device]

    # Iterate epochs
    for t in range(1, epochs):
        d_true = np.linalg.norm(p_true[:, None, :] - p_true[None, :, :], axis=2)
        rssi = rssi_from_distance(d_true, r0=r0, eta=eta, noise_std=rssi_noise_std, rng=rng)

        q = sigmoid((rssi - r_min) / kappa)
        np.fill_diagonal(q, 0.0)

        # Option B: neighbors share refined estimate from previous epoch
        send_pos = p_hat[t - 1]

        for i in range(n_nodes):
            neigh_idx = topk_neighbors_by_quality(q[i], neighbor_budget, min_quality=min_quality)

            if neigh_idx.size == 0:
                if (i == cold_start_device) and (t < cold_start_epochs):
                    p_hat[t, i] = p_hat[t - 1, i]
                    Sigma_hat[t, i] = Sigma_hat[t - 1, i]
                else:
                    p_hat[t, i] = p_raw[t, i]
                    Sigma_hat[t, i] = Sigma_raw[i]
                continue

            rssi_ij = rssi[i, neigh_idx]
            d_hat = distance_from_rssi(rssi_ij, r0=r0, eta=eta)

            q_ij = q[i, neigh_idx]
            sigma_d = sigma_d_base + sigma_d_scale * (1.0 - q_ij)

            # Prior: raw fix for normal nodes, weak prior for cold-start node during early epochs
            if (i == cold_start_device) and (t < cold_start_epochs):
                prior_mean = p_hat[t - 1, i]
                prior_cov = gamma_boot * Sigma_raw[i]
                anchor = prior_mean
            else:
                prior_mean = p_raw[t, i]
                prior_cov = Sigma_raw[i]
                anchor = prior_mean

            if use_trust:
                dist_cons = np.linalg.norm(send_pos[neigh_idx] - anchor, axis=1)
                s = np.exp(-(dist_cons**2) / (2.0 * tau_consistency**2))
                omega = q_ij * s
            else:
                omega = q_ij

            p_i_hat, S_i_hat = refine_with_ranges_gauss_newton(
                prior_mean=prior_mean,
                prior_cov=prior_cov,
                neigh_pos=send_pos[neigh_idx],
                d_hat=d_hat,
                sigma_d=sigma_d,
                omega=omega,
                iters=gn_iters
            )

            p_hat[t, i] = p_i_hat
            Sigma_hat[t, i] = S_i_hat

    err = np.linalg.norm(p_hat - p_true[None, :, :], axis=2)
    return {
        "p_true": p_true,
        "p_hat": p_hat,
        "err": err
    }


# ============================================================
# 100 runs + plots
# ============================================================

def ecdf(x: np.ndarray):
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys


def run_100_runs_cold_start(
    runs: int = 100,
    base_seed: int = 1,
    n_nodes: int = 50,
    epochs: int = 30,
    neighbor_budget: int = 4,
    grid_size: float = 100.0,
    cold_start_device: int = 0,
    cold_start_epochs: int = 10,
    gamma_boot: float = 1000.0,
    use_trust: bool = True,
):
    cold_err_by_epoch = np.zeros((runs, epochs), dtype=float)

    for r in range(runs):
        seed = base_seed + r
        sim = simulate_with_one_cold_start_device(
            n_nodes=n_nodes,
            epochs=epochs,
            neighbor_budget=neighbor_budget,
            grid_size=grid_size,
            seed=seed,
            cold_start_device=cold_start_device,
            cold_start_epochs=cold_start_epochs,
            gamma_boot=gamma_boot,
            use_trust=use_trust,
        )
        cold_err_by_epoch[r, :] = sim["err"][:, cold_start_device]

    e0 = cold_err_by_epoch[:, 0]
    eF = cold_err_by_epoch[:, -1]
    e_cold = cold_err_by_epoch[:, :cold_start_epochs].mean(axis=1)

    improvement_abs = e0 - eF
    improved = np.sum(improvement_abs > 0)

    print(f"Runs: {runs}")
    print(f"Cold-start device: {cold_start_device}")
    print(f"Cold-start epochs: {cold_start_epochs}")
    print(f"Neighbor budget: {neighbor_budget}")
    print(f"gamma_boot: {gamma_boot}")
    print("")
    print(f"Mean error at epoch 0:         {e0.mean():.3f} ± {e0.std(ddof=1):.3f} m")
    print(f"Mean error at final epoch:     {eF.mean():.3f} ± {eF.std(ddof=1):.3f} m")
    print(f"Mean error during cold window: {e_cold.mean():.3f} ± {e_cold.std(ddof=1):.3f} m")
    print(f"Mean improvement (e0 - eF):    {improvement_abs.mean():.3f} ± {improvement_abs.std(ddof=1):.3f} m")
    print(f"Runs improved (final < e0):    {improved}/{runs} ({100.0*improved/runs:.1f}%)")

    # Plot 1: mean + IQR across runs
    t = np.arange(epochs)
    mean_line = cold_err_by_epoch.mean(axis=0)
    p25 = np.percentile(cold_err_by_epoch, 25, axis=0)
    p75 = np.percentile(cold_err_by_epoch, 75, axis=0)

    plt.figure()
    plt.plot(t, mean_line, label="Mean (cold-start device)")
    plt.fill_between(t, p25, p75, alpha=0.25, label="IQR (25–75%)")
    plt.axvline(cold_start_epochs - 1, linestyle="--", label="Cold-start ends")
    plt.xlabel("Epoch")
    plt.ylabel("Localization error (m)")
    plt.title("Cold-start device: error vs epoch across runs")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Plot 2: boxplot epoch0 vs final
    plt.figure()
    plt.boxplot([e0, eF], labels=["Epoch 0", "Final"], showmeans=True)
    plt.ylabel("Localization error (m)")
    plt.title("Cold-start device: epoch 0 vs final (across runs)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Plot 3: ECDF improvement
    xs, ys = ecdf(improvement_abs)
    plt.figure()
    plt.plot(xs, ys)
    plt.axvline(0, linewidth=1)
    plt.xlabel("Improvement (m) = error(epoch0) - error(final)")
    plt.ylabel("Fraction of runs ≤ x")
    plt.title("Cold-start device: ECDF of improvement")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()

    return {
        "cold_err_by_epoch": cold_err_by_epoch,
        "e0": e0,
        "eF": eF,
        "improvement_abs": improvement_abs,
    }


if __name__ == "__main__":
    RUNS = 100
    BASE_SEED = 1
    N_NODES = 50
    EPOCHS = 30
    GRID_SIZE = 100.0
    NEIGHBOR_BUDGET = 4

    COLD_START_DEVICE = 0
    COLD_START_EPOCHS = 10
    GAMMA_BOOT = 1000.0

    results = run_100_runs_cold_start(
        runs=RUNS,
        base_seed=BASE_SEED,
        n_nodes=N_NODES,
        epochs=EPOCHS,
        neighbor_budget=NEIGHBOR_BUDGET,
        grid_size=GRID_SIZE,
        cold_start_device=COLD_START_DEVICE,
        cold_start_epochs=COLD_START_EPOCHS,
        gamma_boot=GAMMA_BOOT,
        use_trust=True,
    )
