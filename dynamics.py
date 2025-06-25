import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: Define Weitz et al. environment-dependent payoff matrix ---
def weitz_matrix_env(n, T=2.0, R=1.5, S=0.5, P=0.1):
    """
    A(n) = (1 - n) * [T P; R S] + n * [R S; T P]
    Equivalently, simplified:
    A(n) = [[T - (T - R)n,     P - (P - S)n],
            [R + (T - R)n,     S + (P - S)n]]
    """
    a11 = T - (T - R) * n
    a12 = P - (P - S) * n
    a21 = R + (T - R) * n
    a22 = S + (P - S) * n
    return np.array([[a11, a12], [a21, a22]])

# --- Step 2: Replicator and Environment Dynamics ---
def replicator_dynamics(x, n, T=2.0, R=1.5, S=0.5, P=0.1, theta=1.0):
    matrix = weitz_matrix_env(n, T, R, S, P)
    f_C = x * matrix[0][0] + (1 - x) * matrix[0][1]
    f_D = x * matrix[1][0] + (1 - x) * matrix[1][1]
    avg_fitness = x * f_C + (1 - x) * f_D
    dx = x * (f_C - avg_fitness)
    x_thresh = 0.5  # threshold cooperation level to sustain environment
    dn = theta * (x - x_thresh)

    return dx, dn

# --- Step 3: Simulation ---
def simulate_weitz_env(T=300, dt=0.01, x0=0.6, n0=0.5, T_val=2.0, R_val=1.5, S_val=0.5, P_val=0.1, theta=1.0):
    steps = int(T / dt)
    x, n = x0, n0
    traj = {'t': [], 'x': [], 'n': []}

    for step in range(steps):
        dx, dn = replicator_dynamics(x, n, T_val, R_val, S_val, P_val, theta)
        x += dx * dt
        n += dn * dt
        x = np.clip(x, 0, 1)
        n = np.clip(n, 0, 1)
        traj['t'].append(step * dt)
        traj['x'].append(x)
        traj['n'].append(n)

    return traj

# --- Step 4: Plot ---
def plot_weitz_env(traj):
    plt.figure(figsize=(10, 6))
    plt.plot(traj['t'], traj['x'], label='Cooperation Level (x)')
    plt.plot(traj['t'], traj['n'], label='Environmental State (n)')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Weitz et al. Replicator Dynamics with Environmental Feedback')
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Run Example ---
if __name__ == "__main__":
    result = simulate_weitz_env(x0=0.6, n0=0.3, T_val=2.0, R_val=1.5, S_val=0.5, P_val=0.5, theta=1.0)
    plot_weitz_env(result)