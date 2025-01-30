"""Newton's method for linear equality-constrained optimization.
"""

import numpy as np
import matplotlib.pyplot as plt

# Objective function and its gradient and Hessian
def f(x):
    return -np.sum(np.log(x))

def grad_f(x):
    return -1 / x

def hess_f(x):
    return np.diag(1 / x**2)

# Newton's method for equality constraints
def newton_method(starting_point, A, b, max_iters=20, tol=1e-6):
    x = np.array(starting_point, dtype=np.float64)
    lambdas = np.zeros(A.shape[0])  # Lagrange multipliers
    trajectory = [x.copy()]

    for _ in range(max_iters):
        grad = grad_f(x)
        hess = hess_f(x)
        
        # Solve the KKT system
        KKT_matrix = np.block([[hess, A.T], [A, np.zeros((A.shape[0], A.shape[0]))]])
        rhs = np.hstack([-grad, b - A @ x])
        step = np.linalg.solve(KKT_matrix, rhs)
        
        dx = step[:len(x)]
        dlambda = step[len(x):]
        
        # Update x and Lagrange multipliers
        x += dx
        lambdas += dlambda
        trajectory.append(x.copy())

        # Check for convergence
        if np.linalg.norm(dx) < tol:
            break

    return x, np.array(trajectory)

# Problem setup
A = np.array([[1, 1]])
b = np.array([10])

# Starting points
start_points = [(9, 1), (5, 5)]

# Solve and plot
plt.figure(figsize=(8, 6))
x1 = np.linspace(1, 10, 100)
x2 = 10 - x1
plt.plot(x1, x2, label="Constraint $x_1 + x_2 = 10$", linestyle="--")

for start in start_points:
    x_opt, trajectory = newton_method(start, A, b)
    trajectory_x1, trajectory_x2 = trajectory[:, 0], trajectory[:, 1]
    
    plt.plot(trajectory_x1, trajectory_x2, marker="o", label=f"Start: {start}")
    plt.scatter(x_opt[0], x_opt[1], marker="*", color="red", s=100, label=f"Optimum ({x_opt[0]:.2f}, {x_opt[1]:.2f})")

# Plot settings
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Convergence Trajectories of Newton's Method")
plt.legend()
plt.grid(True)
plt.show()
