import numpy as np
import matplotlib.pyplot as plt

# --- Define the Weitz et al. environment-dependent payoff matrix ---
def weitz_matrix_env(n, T=2.0, R=1.5, S=0.5, P=0.75):
    a11 = T - (T - R) * n
    a12 = P - (P - S) * n
    a21 = R + (T - R) * n
    a22 = S + (P - S) * n
    return np.array([[a11, a12], [a21, a22]])

# --- Define the dynamics ---
def replicator_dynamics(x, n, T=2.0, R=1.5, S=0.5, P=0.75, theta=1.0, x_thresh=0.5):
    matrix = weitz_matrix_env(n, T, R, S, P)
    f_C = x * matrix[0, 0] + (1 - x) * matrix[0, 1]
    f_D = x * matrix[1, 0] + (1 - x) * matrix[1, 1]
    avg_fitness = x * f_C + (1 - x) * f_D
    dx = x * (f_C - avg_fitness)
    dn = theta * (x - x_thresh)
    return dx, dn

# --- Simulate trajectories for initial conditions ---
def simulate_trajectory(x0, n0, T=2.0, R=1.5, S=0.5, P=0.75, theta=1.0, x_thresh=0.5, dt=0.05, steps=1000):
    x, n = x0, n0
    traj_x, traj_n = [x], [n]
    for _ in range(steps):
        dx, dn = replicator_dynamics(x, n, T, R, S, P, theta, x_thresh)
        x += dx * dt
        n += dn * dt
        x = np.clip(x, 0, 1)
        n = np.clip(n, 0, 1)
        traj_x.append(x)
        traj_n.append(n)
    return traj_x, traj_n

# --- Plot phase plane ---
def plot_phase_plane():
    x_vals = np.linspace(0.01, 0.99, 25)
    n_vals = np.linspace(0.01, 0.99, 25)
    X, N = np.meshgrid(x_vals, n_vals)
    dX, dN = np.zeros_like(X), np.zeros_like(N)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            dx, dn = replicator_dynamics(X[i, j], N[i, j])
            dX[i, j] = dx
            dN[i, j] = dn

    plt.figure(figsize=(10, 7))
    plt.quiver(X, N, dX, dN, color='gray', alpha=0.6, scale=5)

    # Simulate and plot some trajectories
    inits = [(0.2, 0.2), (0.6, 0.3), (0.8, 0.6), (0.3, 0.8), (0.5, 0.5)]
    for x0, n0 in inits:
        tx, tn = simulate_trajectory(x0, n0)
        plt.plot(tx, tn, label=f"start=({x0:.1f},{n0:.1f})")

    plt.xlabel("Cooperation (x)")
    plt.ylabel("Environmental State (n)")
    plt.title("Phase Plane Dynamics (Weitz et al.)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Run ---
if __name__ == "__main__":
    plot_phase_plane()
