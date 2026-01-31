import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple


# --------------------------
# Helpers
# --------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def rssi_from_distance(
    d: np.ndarray,
    r0: float = -40.0,     # RSSI at 1m
    eta: float = 2.0,      # path loss exponent
    d0: float = 1.0,
    # --- Gaussian mixture (LOS/NLOS) params ---
    p_nlos: float = 0.20,      # probability of NLOS / outlier
    sigma_los: float = 2.0,    # LOS noise std (dB)
    mu_nlos: float = -8.0,     # NLOS bias (dB)
    sigma_nlos: float = 6.0,   # NLOS noise std (dB)
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Log-distance path loss model for RSSI with Gaussian mixture noise:
      with prob (1-p_nlos): N(0, sigma_los^2)   [LOS]
      with prob p_nlos:     N(mu_nlos, sigma_nlos^2)  [NLOS/outlier]
    """
    if rng is None:
        rng = np.random.default_rng()

    d_safe = np.maximum(d, 1e-3)
    mean_rssi = r0 - 10.0 * eta * np.log10(d_safe / d0)

    is_nlos = rng.random(size=d_safe.shape) < p_nlos

    noise = np.empty(d_safe.shape, dtype=float)
    noise[~is_nlos] = rng.normal(loc=0.0, scale=sigma_los, size=np.count_nonzero(~is_nlos))
    noise[is_nlos] = rng.normal(loc=mu_nlos, scale=sigma_nlos, size=np.count_nonzero(is_nlos))

    return mean_rssi + noise


def distance_from_rssi(rssi: np.ndarray, r0: float = -40.0, eta: float = 2.0, d0: float = 1.0) -> np.ndarray:
    """Invert log-distance model to get distance estimate from RSSI."""
    return d0 * (10.0 ** ((r0 - rssi) / (10.0 * eta)))


def topk_neighbors_by_quality(q_row: np.ndarray, k: int, min_quality: float = 0.05) -> np.ndarray:
    """Pick up to k neighbors with the highest link quality."""
    idx = np.where(q_row >= min_quality)[0]
    if idx.size == 0:
        return np.array([], dtype=int)
    idx_sorted = idx[np.argsort(q_row[idx])[::-1]]
    return idx_sorted[:k]


# --------------------------
# Gauss-Newton refinement using range constraints
# --------------------------

def refine_with_ranges_gauss_newton(
    p0: np.ndarray,
    Sigma_raw: np.ndarray,
    neigh_pos: np.ndarray,
    d_hat: np.ndarray,
    sigma_d: np.ndarray,
    omega: np.ndarray,
    iters: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve:
      min_p  (p - p_raw)^T Sigma_raw^-1 (p - p_raw)
           + sum_j omega_j * ((||p - p_j|| - d_hat_j)^2 / sigma_d_j^2)
    """
    W_prior = np.linalg.inv(Sigma_raw)
    p = p0.copy()

    for _ in range(iters):
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
            w = (wj / (sdj**2))

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


# --------------------------
# Simulation (returns mean error per epoch)
# --------------------------

def simulate_mean_error_by_epoch(
    n_nodes: int,
    epochs: int,
    neighbor_budget: int,
    grid_size: float,
    seed: int,
    # Mixture params
    p_nlos: float,
    sigma_los: float,
    mu_nlos: float,
    sigma_nlos: float
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    # True positions
    p_true = rng.uniform(0.0, grid_size, size=(n_nodes, 2))

    # Heterogeneous GNSS-like raw covariance per node
    sigma_x = rng.uniform(3.0, 12.0, size=n_nodes)
    sigma_y = rng.uniform(3.0, 12.0, size=n_nodes)
    Sigma_raw = np.array([np.diag([sx**2, sy**2]) for sx, sy in zip(sigma_x, sigma_y)])

    # Raw fixes (per epoch) + refined estimates
    p_raw = np.zeros((epochs, n_nodes, 2))
    p_hat = np.zeros((epochs, n_nodes, 2))
    Sigma_hat = np.zeros((epochs, n_nodes, 2, 2))

    # Link-quality parameters
    r_min = -85.0
    kappa = 5.0
    min_quality = 0.10

    # Consistency gating
    tau_consistency = 25.0

    # RSSI model parameters (shared with inversion)
    r0 = -40.0
    eta = 2.0

    # Generate raw measurements each epoch
    for t in range(epochs):
        for i in range(n_nodes):
            p_raw[t, i] = p_true[i] + rng.multivariate_normal([0.0, 0.0], Sigma_raw[i])

    # Epoch 0: initial fix = raw
    p_hat[0] = p_raw[0]
    for i in range(n_nodes):
        Sigma_hat[0, i] = Sigma_raw[i]

    # True distances (static nodes)
    d_true = np.linalg.norm(p_true[:, None, :] - p_true[None, :, :], axis=2)

    # Iterate
    for t in range(1, epochs):
        rssi = rssi_from_distance(
            d_true,
            r0=r0,
            eta=eta,
            p_nlos=p_nlos,
            sigma_los=sigma_los,
            mu_nlos=mu_nlos,
            sigma_nlos=sigma_nlos,
            rng=rng
        )

        q = sigmoid((rssi - r_min) / kappa)
        np.fill_diagonal(q, 0.0)

        # Option B: neighbors share refined estimate from previous epoch
        send_pos = p_hat[t - 1]

        for i in range(n_nodes):
            neigh_idx = topk_neighbors_by_quality(q[i], neighbor_budget, min_quality=min_quality)

            if neigh_idx.size == 0:
                p_hat[t, i] = p_raw[t, i]
                Sigma_hat[t, i] = Sigma_raw[i]
                continue

            rssi_ij = rssi[i, neigh_idx]
            d_hat = distance_from_rssi(rssi_ij, r0=r0, eta=eta)

            q_ij = q[i, neigh_idx]
            sigma_d = 2.0 + 10.0 * (1.0 - q_ij)

            dist_cons = np.linalg.norm(send_pos[neigh_idx] - p_raw[t, i], axis=1)
            s = np.exp(-(dist_cons**2) / (2.0 * tau_consistency**2))
            omega = q_ij * s

            p_i_hat, S_i_hat = refine_with_ranges_gauss_newton(
                p0=p_raw[t, i],
                Sigma_raw=Sigma_raw[i],
                neigh_pos=send_pos[neigh_idx],
                d_hat=d_hat,
                sigma_d=sigma_d,
                omega=omega,
                iters=6
            )

            p_hat[t, i] = p_i_hat
            Sigma_hat[t, i] = S_i_hat

    err = np.linalg.norm(p_hat - p_true[None, :, :], axis=2)  # (epochs, n_nodes)
    return err.mean(axis=1)


# --------------------------
# Run budgets 1..25 (100 runs each) and plot
# --------------------------

def run_neighbor_budget_sweep(
    n_nodes: int = 50,
    max_budget: int = 25,
    epochs: int = 30,
    grid_size: float = 100.0,
    runs: int = 100,
    base_seed: int = 1,
    # Mixture params
    p_nlos: float = 0.25,
    sigma_los: float = 2.0,
    mu_nlos: float = -10.0,
    sigma_nlos: float = 7.0
) -> Dict[str, np.ndarray]:
    budgets = np.arange(1, max_budget + 1)
    B = budgets.size

    # store mean error per epoch for each budget and run: (B, runs, epochs)
    err_bre = np.zeros((B, runs, epochs), dtype=float)

    for bi, b in enumerate(budgets):
        for r in range(runs):
            seed = base_seed + 10000 * int(b) + r
            err_bre[bi, r, :] = simulate_mean_error_by_epoch(
                n_nodes=n_nodes,
                epochs=epochs,
                neighbor_budget=int(b),
                grid_size=grid_size,
                seed=seed,
                p_nlos=p_nlos,
                sigma_los=sigma_los,
                mu_nlos=mu_nlos,
                sigma_nlos=sigma_nlos
            )
        print(f"budget={int(b):2d}/{max_budget} | final mean error = {err_bre[bi, :, -1].mean():.3f} m")

    mean_err = err_bre.mean(axis=1)               # (B, epochs)
    std_err = err_bre.std(axis=1, ddof=1)         # (B, epochs)

    return {"budgets": budgets, "mean_err": mean_err, "std_err": std_err, "raw": err_bre}


def plot_budget_sweep(res: Dict[str, np.ndarray], epochs: int) -> None:
    budgets = res["budgets"]
    mean_err = res["mean_err"]
    std_err = res["std_err"]

    # Final epoch error vs budget
    plt.figure()
    plt.errorbar(budgets, mean_err[:, -1], yerr=std_err[:, -1], capsize=3)
    plt.xlabel("Neighbor budget n")
    plt.ylabel("Mean localization error at final epoch (m)")
    plt.title("Neighbor budget sweep (1..25): final error (mean ± std over runs)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Optional: plot error vs budget for a few epochs
    selected_epochs = [0, min(5, epochs - 1), min(10, epochs - 1), epochs - 1]
    plt.figure()
    for t in selected_epochs:
        plt.plot(budgets, mean_err[:, t], label=f"Epoch {t}")
    plt.xlabel("Neighbor budget n")
    plt.ylabel("Mean localization error (m)")
    plt.title("Neighbor budget sweep: error vs budget at selected epochs")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Heatmap: budget vs epoch
    plt.figure()
    im = plt.imshow(
        mean_err,
        aspect="auto",
        origin="lower",
        extent=[0, epochs - 1, budgets[0], budgets[-1]]
    )
    plt.colorbar(im, label="Mean localization error (m)")
    plt.xlabel("Epoch")
    plt.ylabel("Neighbor budget n")
    plt.title("Mean error heatmap (budgets 1..25, averaged over runs)")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    # ---- Experiment parameters ----
    N_NODES = 50
    MAX_BUDGET = 49
    EPOCHS = 30
    GRID_SIZE = 100.0
    RUNS = 100
    BASE_SEED = 1

    # ---- RSSI mixture params ----
    P_NLOS = 0.25
    SIG_LOS = 2.0
    MU_NLOS = -10.0
    SIG_NLOS = 7.0

    res = run_neighbor_budget_sweep(
        n_nodes=N_NODES,
        max_budget=MAX_BUDGET,
        epochs=EPOCHS,
        grid_size=GRID_SIZE,
        runs=RUNS,
        base_seed=BASE_SEED,
        p_nlos=P_NLOS,
        sigma_los=SIG_LOS,
        mu_nlos=MU_NLOS,
        sigma_nlos=SIG_NLOS
    )

    plot_budget_sweep(res, epochs=EPOCHS)
