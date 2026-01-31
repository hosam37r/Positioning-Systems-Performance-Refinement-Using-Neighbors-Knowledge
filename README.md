# Neighbor-Assisted Cooperative Positioning Refinement

This repository contains a Python-based simulation environment for **Decentralized Neighbor-Assisted Localization**. The framework models a network of smartphones that improve their positioning accuracy by exchanging compact location summaries and forming relative proximity constraints (e.g., via RSSI) with nearby devices.

The code implements an **Uncertainty-Aware** and **Trust-Aware** refinement algorithm using Gauss-Newton optimization, designed to address specific research questions regarding accuracy, convergence speed, security, and scalability.

## 📋 Table of Contents
- [Overview](#overview)
- [Research Questions](#research-questions)
- [Repository Structure](#repository-structure)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Key Parameters](#key-parameters)

## 🔭 Overview

In urban environments, standalone GNSS positioning is often degraded. This project simulates a cooperative approach where devices:
1.  **Obtain** a local fix (e.g., GNSS) with estimated uncertainty covariance.
2.  **Discover** neighbors and estimate link quality (RSSI).
3.  **Exchange** refined location estimates (single-hop).
4.  **Refine** their position using a weighted non-linear least squares objective that fuses the local prior with neighbor range constraints.

The simulation handles heterogeneous device capabilities, distance-dependent noise, outlier links (NLOS), and malicious actors.

## ❓ Research Questions

The simulation scripts are organized to address four specific research questions:

1.  **RQ1 (Accuracy):** Can a device improve its positioning accuracy by incorporating neighbors' location estimates using uncertainty-aware weighting?
2.  **RQ2 (Convergence):** Can cooperative neighbor information reduce time-to-fix (TTF) and accelerate convergence, specifically for devices with poor initial fixes (cold start)?
3.  **RQ3 (Robustness):** How can neighborhood exchange be used to identify and mitigate the influence of unreliable or malicious nodes (spoofing)?
4.  **RQ4 (Scalability):** How does the number of participating neighbors (budget $n$) affect accuracy, computational cost, and overall system performance?

## 📂 Repository Structure

| File | Description | Associated RQ |
| :--- | :--- | :--- |
| `main.py` | Core simulation engine. Implements the baseline uncertainty-aware Gauss-Newton refinement and validates general accuracy improvements. | **RQ1** |
| `rq2.py` | Simulates "Cold Start" scenarios. Uses covariance inflation (`gamma_boot`) to prioritize neighbor constraints when local confidence is low. | **RQ2** |
| `rq3.py` | Implements the **Trust/Consistency Scoring** mechanism. Simulates malicious nodes (spoofers) and evaluates the algorithm's ability to flag and down-weight them. | **RQ3** |
| `rq4.py` | Scaling study focusing on **Accuracy**. Sweeps neighbor budgets ($n=1..N$) to observe error reduction saturation. | **RQ4** |
| `rq4-2.py` | Scaling study focusing on **Computational Cost**. Analyzes the tradeoff between error reduction and compute units (iterations $\times$ neighbors). | **RQ4** |

## 🧠 Methodology

The core algorithm is a decentralized refinement process executed at every epoch $t$ by every device $i$.

### 1. Refinement Objective
Devices solve a localized optimization problem to minimize the residual between their local prior and observations from neighbors:

$$
\hat{\mathbf{p}}_i = \arg\min_{\mathbf{p}} \left[ (\mathbf{p} - \tilde{\mathbf{p}}_i)^T \tilde{\mathbf{\Sigma}}_i^{-1} (\mathbf{p} - \tilde{\mathbf{p}}_i) + \sum_{j \in \mathcal{N}_i} \omega_{ij} \frac{(\|\mathbf{p} - \hat{\mathbf{p}}_j\| - \hat{d}_{ij})^2}{\sigma_{d,ij}^2} \right]
$$

Where:
* $\tilde{\mathbf{p}}_i, \tilde{\mathbf{\Sigma}}_i$: Local GNSS fix and covariance.
* $\hat{\mathbf{p}}_j$: Position estimate received from neighbor $j$.
* $\hat{d}_{ij}$: Estimated distance to neighbor (derived from RSSI).
* $\omega_{ij}$: Weight combining link quality and trust.

### 2. Trust and Consistency (RQ3)
To mitigate malicious actors, we compute a normalized range residual $\epsilon_{ij}$ between the observed range and the reported position. A trust score $s_{ij}$ is calculated and smoothed via an Exponential Moving Average (EMA):

$$
s_{ij}(t) = \exp\left(-\frac{\epsilon_{ij}^2(t)}{2\lambda^2}\right)
$$

Neighbors with low consistency scores are flagged and their weights $\omega_{ij}$ are set to zero.

### 3. Cooperative Bootstrapping (RQ2)
For devices in "cold start" (unstable local fix), the algorithm temporarily inflates the local covariance ($\tilde{\mathbf{\Sigma}}_i \leftarrow \gamma_{\text{boot}} \tilde{\mathbf{\Sigma}}_i$), forcing the solver to rely more heavily on neighbor constraints to accelerate convergence.

## ⚙️ Installation

The simulations are built on standard Python scientific computing libraries.

```bash
# Clone the repository
git clone [https://github.com/yourusername/cooperative-positioning.git](https://github.com/yourusername/cooperative-positioning.git)
cd cooperative-positioning

# Install dependencies
pip install numpy matplotlib
