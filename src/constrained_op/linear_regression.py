"""
Solve and plotlinear regression with different norms using CVXPY.
"""

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

# Generate a synthetic dataset
theta1_act = 2  # Actual slope
theta2_act = 5  # Actual intercept
N = 200  # Number of points
x = np.arange(0, N)

# Design matrix
X = np.vstack([x, np.ones(N)]).T

# Random number generator
rng = np.random.default_rng(seed=42)


def solve_and_plot(noise_mag, ax):
    # Generate dataset
    y = theta1_act * x + theta2_act + rng.normal(0, noise_mag, N)

    # Solve l2 regression
    Theta_l2 = np.linalg.inv(X.T @ X) @ X.T @ y
    theta1_l2, theta2_l2 = Theta_l2

    # Solve l1 regression
    Theta = cp.Variable(2)
    t = cp.Variable(N)
    problem_l1 = cp.Problem(
        cp.Minimize(cp.sum(t)), [y - X @ Theta <= t, y - X @ Theta >= -t]
    )
    problem_l1.solve()
    theta1_l1, theta2_l1 = Theta.value

    # Solve l_infinity regression
    Theta = cp.Variable(2)
    t = cp.Variable()
    problem_linf = cp.Problem(cp.Minimize(t), [y - X @ Theta <= t, y - X @ Theta >= -t])
    problem_linf.solve()
    theta1_linf, theta2_linf = Theta.value

    # Plot
    ax.scatter(x, y, label="Data", alpha=0.5)
    x_vals = np.linspace(min(x), max(x), 100)

    ax.plot(
        x_vals,
        theta1_l2 * x_vals + theta2_l2,
        label=f"L2: Y ~ {theta1_l2:.2f}X + {theta2_l2:.2f}",
        color="blue",
        linewidth=2,
    )
    ax.plot(
        x_vals,
        theta1_l1 * x_vals + theta2_l1,
        label=f"L1: Y ~ {theta1_l1:.2f}X + {theta2_l1:.2f}",
        color="red",
        linewidth=2,
    )
    ax.plot(
        x_vals,
        theta1_linf * x_vals + theta2_linf,
        label=f"L-inf: Y ~ {theta1_linf:.2f}X + {theta2_linf:.2f}",
        color="green",
        linewidth=2,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Noise = {noise_mag}", fontsize=12)
    ax.legend()
    ax.grid()


# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 10))
solve_and_plot(30, ax1)
solve_and_plot(60, ax2)
plt.suptitle("Regression Lines with Different Norms and Noise Levels", fontsize=12)
plt.tight_layout()
plt.show()
