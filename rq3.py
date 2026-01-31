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
    """Log-distance path-loss RSSI model."""
    if rng is None:
        rng = np.random.default_rng()
    d_safe = np.maximum(d, 1e-3)
    return r0 - 10.0 * eta * np.log10(d_safe / d0) + rng.normal(0.0, noise_std, size=d_safe.shape)


def distance_from_rssi(rssi: np.ndarray, r0: float = -40.0, eta: float = 2.0, d0: float = 1.0) -> np.ndarray:
    """Invert log-distance model: RSSI -> distance estimate."""
    return d0 * (10.0 ** ((r0 - rssi) / (10.0 * eta)))


def topk_neighbors_by_quality(q_row: np.ndarray, k: int, min_quality: float) -> np.ndarray:
    """Pick up to k neighbors with highest link quality."""
    candidates = np.where(q_row >= min_quality)[0]
    if candidates.size == 0:
        return np.array([], dtype=int)
    order = np.argsort(q_row[candidates])[::-1]
    return candidates[order][:k]


# ============================================================
# Gauss-Newton refinement (range constraints)
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

    Returns refined p and approx covariance (~inv Hessian).
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
# Trust scoring (RQ3): epsilon_ij, s_ij, EMA smoothing
# ============================================================

def trust_score_from_range_residual(
    p_raw_i: np.ndarray,
    p_hat_j: np.ndarray,
    d_hat_ij: float,
    sigma_d_ij: float,
    lam: float
) -> float:
    """
    epsilon_ij = | ||p_raw_i - p_hat_j|| - d_hat_ij | / sigma_d_ij
    s_ij = exp( - epsilon_ij^2 / (2 * lam^2) )
    """
    eps = abs(np.linalg.norm(p_raw_i - p_hat_j) - d_hat_ij) / max(sigma_d_ij, 1e-9)
    s = np.exp(-(eps ** 2) / (2.0 * (lam ** 2)))
    return float(s)


# ============================================================
# Simulation with malicious nodes + trust mitigation
# ============================================================

def simulate_rq3_trust(
    n_nodes: int = 50,
    epochs: int = 30,
    neighbor_budget: int = 4,
    grid_size: float = 100.0,
    seed: int = 1,
    malicious_frac: float = 0.2,   # 20% attackers
    spoof_mode: str = "random",    # "random" or "offset"
    spoof_offset_std: float = 40.0,
    # Link params
    r0: float = -40.0,
    eta: float = 2.0,
    rssi_noise_std: float = 2.0,
    r_min: float = -85.0,
    kappa: float = 5.0,
    min_quality: float = 0.10,
    # Range uncertainty
    sigma_d_base: float = 2.0,
    sigma_d_scale: float = 10.0,
    # Trust params (your equations)
    use_trust: bool = True,
    lam_trust: float = 2.0,     # lambda in exp(-eps^2/(2 lambda^2))
    eta_ema: float = 0.8,       # EMA smoothing factor
    # GN
    gn_iters: int = 6
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    # True positions
    p_true = rng.uniform(0.0, grid_size, size=(n_nodes, 2))

    # Heterogeneous local covariances (raw fix quality)
    sigma_x = rng.uniform(3.0, 12.0, size=n_nodes)
    sigma_y = rng.uniform(3.0, 12.0, size=n_nodes)
    Sigma_raw = np.array([np.diag([sx**2, sy**2]) for sx, sy in zip(sigma_x, sigma_y)])

    # Choose malicious nodes
    m = int(round(malicious_frac * n_nodes))
    malicious = np.zeros(n_nodes, dtype=bool)
    if m > 0:
        malicious_idx = rng.choice(n_nodes, size=m, replace=False)
        malicious[malicious_idx] = True
    else:
        malicious_idx = np.array([], dtype=int)

    # Raw fixes and refined estimates
    p_raw = np.zeros((epochs, n_nodes, 2))
    p_hat = np.zeros((epochs, n_nodes, 2))
    Sigma_hat = np.zeros((epochs, n_nodes, 2, 2))

    # Trust EMA state: bar_s_ij
    bar_s = np.ones((n_nodes, n_nodes), dtype=float)
    # Track which links were actually used (for analysis)
    used_count = np.zeros((n_nodes, n_nodes), dtype=int)

    # Generate raw fixes each epoch
    for t in range(epochs):
        for i in range(n_nodes):
            p_raw[t, i] = p_true[i] + rng.multivariate_normal([0.0, 0.0], Sigma_raw[i])

    # Epoch 0 init
    p_hat[0] = p_raw[0]
    for i in range(n_nodes):
        Sigma_hat[0, i] = Sigma_raw[i]

    # Iterate epochs
    for t in range(1, epochs):
        # True distances -> simulate RSSI
        d_true = np.linalg.norm(p_true[:, None, :] - p_true[None, :, :], axis=2)
        rssi = rssi_from_distance(d_true, r0=r0, eta=eta, noise_std=rssi_noise_std, rng=rng)

        # Link quality
        q = sigmoid((rssi - r_min) / kappa)
        np.fill_diagonal(q, 0.0)

        # Option B: neighbors broadcast refined estimate from previous epoch
        send_pos = p_hat[t - 1].copy()

        # Malicious broadcast: spoof their reported positions
        if spoof_mode == "random":
            # every epoch, malicious nodes broadcast random coordinates
            send_pos[malicious] = rng.uniform(0.0, grid_size, size=(malicious.sum(), 2))
        elif spoof_mode == "offset":
            # malicious nodes broadcast true position + large random offset
            offsets = rng.normal(0.0, spoof_offset_std, size=(malicious.sum(), 2))
            send_pos[malicious] = p_true[malicious] + offsets
        else:
            raise ValueError("spoof_mode must be 'random' or 'offset'")

        for i in range(n_nodes):
            neigh_idx = topk_neighbors_by_quality(q[i], neighbor_budget, min_quality=min_quality)

            if neigh_idx.size == 0:
                p_hat[t, i] = p_raw[t, i]
                Sigma_hat[t, i] = Sigma_raw[i]
                continue

            # RSSI -> range estimate
            rssi_ij = rssi[i, neigh_idx]
            d_hat = distance_from_rssi(rssi_ij, r0=r0, eta=eta)

            # Range uncertainty: weaker links noisier
            q_ij = q[i, neigh_idx]
            sigma_d = sigma_d_base + sigma_d_scale * (1.0 - q_ij)

            # Compute trust scores (RQ3 equations) + EMA
            if use_trust:
                s_now = np.zeros(neigh_idx.size, dtype=float)
                for k, j in enumerate(neigh_idx):
                    s_ij = trust_score_from_range_residual(
                        p_raw_i=p_raw[t, i],
                        p_hat_j=send_pos[j],
                        d_hat_ij=float(d_hat[k]),
                        sigma_d_ij=float(sigma_d[k]),
                        lam=lam_trust
                    )
                    # EMA smoothing: bar_s_ij(t) = eta * bar_s_ij(t-1) + (1-eta) * s_ij(t)
                    bar_s[i, j] = eta_ema * bar_s[i, j] + (1.0 - eta_ema) * s_ij
                    s_now[k] = bar_s[i, j]
                    used_count[i, j] += 1

                omega = q_ij * s_now
            else:
                # No trust mitigation baseline
                omega = q_ij
                for j in neigh_idx:
                    used_count[i, j] += 1  # still count used links

            # Refine using GN: prior at current raw fix + range constraints to neighbors
            p_i_hat, S_i_hat = refine_with_ranges_gauss_newton(
                prior_mean=p_raw[t, i],
                prior_cov=Sigma_raw[i],
                neigh_pos=send_pos[neigh_idx],
                d_hat=d_hat,
                sigma_d=sigma_d,
                omega=omega,
                iters=gn_iters
            )

            p_hat[t, i] = p_i_hat
            Sigma_hat[t, i] = S_i_hat

    err = np.linalg.norm(p_hat - p_true[None, :, :], axis=2)  # (epochs, n_nodes)

    return {
        "p_true": p_true,
        "p_raw": p_raw,
        "p_hat": p_hat,
        "err": err,
        "malicious": malicious,
        "bar_s": bar_s,
        "used_count": used_count
    }


# ============================================================
# Compare "with trust" vs "without trust" and plot
# ============================================================

def compare_trust_vs_no_trust(
    runs: int = 100,
    base_seed: int = 1,
    n_nodes: int = 50,
    epochs: int = 30,
    neighbor_budget: int = 4,
    grid_size: float = 100.0,
    malicious_frac: float = 0.2,
    spoof_mode: str = "random",
    lam_trust: float = 2.0,
    eta_ema: float = 0.8
) -> None:
    mean_err_trust = np.zeros((runs, epochs), dtype=float)
    mean_err_no = np.zeros((runs, epochs), dtype=float)

    # Collect trust values on actually-used links (honest->honest vs honest->malicious)
    trust_hh = []
    trust_hm = []

    for r in range(runs):
        seed = base_seed + r

        sim_trust = simulate_rq3_trust(
            n_nodes=n_nodes, epochs=epochs, neighbor_budget=neighbor_budget, grid_size=grid_size,
            seed=seed, malicious_frac=malicious_frac, spoof_mode=spoof_mode,
            use_trust=True, lam_trust=lam_trust, eta_ema=eta_ema
        )
        sim_no = simulate_rq3_trust(
            n_nodes=n_nodes, epochs=epochs, neighbor_budget=neighbor_budget, grid_size=grid_size,
            seed=seed, malicious_frac=malicious_frac, spoof_mode=spoof_mode,
            use_trust=False, lam_trust=lam_trust, eta_ema=eta_ema
        )

        # Evaluate mean error over HONEST nodes (common metric for RQ3)
        honest = ~sim_trust["malicious"]
        mean_err_trust[r, :] = sim_trust["err"][:, honest].mean(axis=1)
        mean_err_no[r, :] = sim_no["err"][:, honest].mean(axis=1)

        # Extract trust values for used links at end of run
        bar_s = sim_trust["bar_s"]
        used = sim_trust["used_count"] > 0
        mal = sim_trust["malicious"]

        # Consider only links from honest i
        for i in range(n_nodes):
            if mal[i]:
                continue
            for j in range(n_nodes):
                if not used[i, j]:
                    continue
                if mal[j]:
                    trust_hm.append(bar_s[i, j])
                else:
                    trust_hh.append(bar_s[i, j])

    # ---- Print summary ----
    e0_t = mean_err_trust[:, 0]
    eF_t = mean_err_trust[:, -1]
    e0_n = mean_err_no[:, 0]
    eF_n = mean_err_no[:, -1]

    print("=== RQ3: Trust mitigation vs no trust (mean error over HONEST nodes) ===")
    print(f"Runs: {runs}, N={n_nodes}, grid={grid_size}x{grid_size}, neighbor_budget={neighbor_budget}")
    print(f"Malicious fraction: {malicious_frac:.2f}, spoof_mode={spoof_mode}")
    print("")
    print(f"WITH trust:    epoch0 {e0_t.mean():.3f} ± {e0_t.std(ddof=1):.3f} m | final {eF_t.mean():.3f} ± {eF_t.std(ddof=1):.3f} m")
    print(f"NO trust:      epoch0 {e0_n.mean():.3f} ± {e0_n.std(ddof=1):.3f} m | final {eF_n.mean():.3f} ± {eF_n.std(ddof=1):.3f} m")
    print(f"Final gain (no_trust - trust): {(eF_n.mean()-eF_t.mean()):.3f} m (positive means trust helps)")

    trust_hh = np.array(trust_hh, dtype=float)
    trust_hm = np.array(trust_hm, dtype=float)
    if trust_hh.size > 0 and trust_hm.size > 0:
        print("")
        print("=== Trust score sanity check (final EMA trust on USED links) ===")
        print(f"Honest->Honest mean trust:   {trust_hh.mean():.3f} ± {trust_hh.std(ddof=1):.3f}  (n={trust_hh.size})")
        print(f"Honest->Malicious mean trust:{trust_hm.mean():.3f} ± {trust_hm.std(ddof=1):.3f}  (n={trust_hm.size})")

    # ---- Plots ----
    t = np.arange(epochs)

    plt.figure()
    plt.plot(t, mean_err_trust.mean(axis=0), label="With trust")
    plt.plot(t, mean_err_no.mean(axis=0), label="No trust")
    plt.xlabel("Epoch")
    plt.ylabel("Mean error over honest nodes (m)")
    plt.title("RQ3: Error vs epoch (trust mitigation vs baseline)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if trust_hh.size > 0 and trust_hm.size > 0:
        plt.figure()
        plt.hist(trust_hh, bins=25, alpha=0.6, label="Honest→Honest")
        plt.hist(trust_hm, bins=25, alpha=0.6, label="Honest→Malicious")
        plt.xlabel("Final EMA trust score  $\overline{s}_{ij}$")
        plt.ylabel("Count (used links)")
        plt.title("Trust score distribution (should separate honest vs malicious)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    # ---- Edit these parameters ----
    RUNS = 100
    N_NODES = 50
    EPOCHS = 30
    GRID_SIZE = 100.0
    NEIGHBOR_BUDGET = 4

    MALICIOUS_FRAC = 0.2
    SPOOF_MODE = "random"     # try "offset" too

    LAMBDA_TRUST = 2.0
    ETA_EMA = 0.8

    compare_trust_vs_no_trust(
        runs=RUNS,
        base_seed=1,
        n_nodes=N_NODES,
        epochs=EPOCHS,
        neighbor_budget=NEIGHBOR_BUDGET,
        grid_size=GRID_SIZE,
        malicious_frac=MALICIOUS_FRAC,
        spoof_mode=SPOOF_MODE,
        lam_trust=LAMBDA_TRUST,
        eta_ema=ETA_EMA
    )
