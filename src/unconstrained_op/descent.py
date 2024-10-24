"""Module for descent methods, including gradient descent, noisy gradient descent, momentum gradient descent, conjugate gradient descent

1. gradient_descent: Gradient descent optimization algorithm.
2. noisy_gradient_descent: Noisy gradient descent optimization algorithm.
3. momentum_gradient_descent: Momentum gradient descent optimization algorithm.
4. conjugate_gradient_descent: Conjugate gradient descent optimization algorithm.
5. plot_steps: Plot the steps taken by the optimization algorithm on a contour plot of the function.
"""

from collections.abc import Callable
from enum import StrEnum, auto, IntEnum
import logging

import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad
from numpy.typing import ArrayLike
from matplotlib.axes import Axes

from .line_search import bisection_method, backtracking_line_search

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.setHandler(logging.FileHandler("logs/descent.log"))


class StepSize(StrEnum):
    FIXED = auto()
    INEXACT = auto()
    EXACT = auto()
    SHRINKING = auto()


class TerminationCondition(IntEnum):
    ABSOLUTE_IMPROVEMENT = auto()
    RELATIVE_IMPROVEMENT = auto()
    GRAD_NORM = auto()
    MAX_ITERATIONS = auto()


def _validate_inputs(func, tol, max_iter, step_size):
    if not callable(func):
        raise ValueError("func must be a callable function")

    if not isinstance(tol, (float, np.floating)):
        raise ValueError(f"Tolerance must be a float. Current type: {type(tol)}")
    elif not (0.0 < tol < 1.0):
        raise ValueError(
            f"Tolerence must be a float between 0 and 1. Current value: {tol}"
        )

    if not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError(
            f"max_iter must be a positive integer. Current type: {type(max_iter)}; current value: {max_iter}"
        )

    valid_stepsize_strategies = [e.value for e in StepSize]
    if step_size not in valid_stepsize_strategies:
        raise ValueError(
            f"step_size must be a valid strategy: {valid_stepsize_strategies}"
        )


def choose_step_size(
    func: Callable,
    step_size: str,
    initial_alpha: float,
    x_i: ArrayLike,
    d_i: ArrayLike,
    i: int,
) -> float:
    if step_size == StepSize.FIXED:
        return initial_alpha
    elif step_size == StepSize.EXACT:
        return bisection_method(func, 0, 1)
    elif step_size == StepSize.INEXACT:
        return backtracking_line_search(func, x_i, d=-d_i)
    elif step_size == StepSize.SHRINKING:
        return initial_alpha / (1 + i)


def check_termination(termination_conditions, i, x_i, d_i, x_next, tol, max_iter):
    if TerminationCondition.GRAD_NORM in termination_conditions:
        if np.linalg.norm(d_i) < tol:
            return True

    if TerminationCondition.ABSOLUTE_IMPROVEMENT in termination_conditions:
        if np.linalg.norm(x_next - x_i) < tol:
            return True

    if TerminationCondition.RELATIVE_IMPROVEMENT in termination_conditions:
        if np.linalg.norm(x_next - x_i) < tol * max(np.linalg.norm(x_i), 1e-8):
            return True

    if TerminationCondition.MAX_ITERATIONS in termination_conditions and i >= max_iter:
        return True

    return False


def gradient_descent(
    func: Callable,
    x0: ArrayLike,
    step_size: StepSize | str = StepSize.FIXED,
    initial_alpha: float = 0.01,
    termination_conditions: list[TerminationCondition] = [
        TerminationCondition.GRAD_NORM,
        TerminationCondition.MAX_ITERATIONS,
    ],
    tol: float = 1e-7,
    max_iter: int = 100,
) -> tuple[ArrayLike, list[ArrayLike], int]:
    """Gradient descent optimization algorithm.

    Args:
        func: callable. Function to be optimized.
        x0: array-like. Initial guess.
        step_size: step size strategy. Valid values are from the StepSize Enum, including 'fixed', 'inexact', 'exact', 'shrinking'.
        initial_alpha: float. Initial step size.
        tol: float. Tolerance to stop the optimization.
        max_iter: int. Maximum number of iterations.

    Returns:
        x_next: array-like. Optimal solution found.
        steps: list of array-like. Steps taken by the algorithm.
        num_steps: int. Number of steps taken.
    """
    _validate_inputs(func, tol, max_iter, step_size)

    x_i = np.array(x0)
    gradient = grad(func)
    steps = [x_i.copy()]  # Store steps for plotting

    i = 1
    while True:
        # Direction d_i and step size alpha
        d_i = -gradient(x_i) / np.linalg.norm(gradient(x_i))
        alpha = choose_step_size(func, step_size, initial_alpha, x_i, d_i, i)
        x_next = x_i - alpha * d_i  # Update formula
        steps.append(x_next.copy())

        if check_termination(
            termination_conditions, i, x_i, d_i, x_next, tol, max_iter
        ):
            # Log experiment conditions and results to a file
            termination_conditions_str = ", ".join(
                [tc.value for tc in termination_conditions]
            )
            step_strategy = step_size if isinstance(step_size, str) else step_size.value

            logger.info(
                f"Gradient descent \t {x_next} \t {x0} \t {step_strategy} \t {termination_conditions_str} \t {tol} \t {i}"
            )

            return x_next, steps, i

        # Update for next iteration
        i += 1
        x_i = x_next


def noisy_gradient_descent(
    func: Callable,
    x0: ArrayLike,
    step_size: StepSize | str = StepSize.FIXED,
    initial_alpha: float = 0.01,
    termination_conditions: list[TerminationCondition] = [
        TerminationCondition.GRAD_NORM,
        TerminationCondition.MAX_ITERATIONS,
    ],
    tol: float = 1e-7,
    max_iter: int = 100,
) -> tuple[ArrayLike, list[ArrayLike], int]:
    """Noisy gradient descent optimization algorithm, adding noise to the update formula

    Args:
        func: callable. Function to be optimized.
        x0: array-like. Initial guess.
        step_size: step size strategy. Valid values are from the StepSize Enum, including 'fixed', 'inexact', 'exact', 'shrinking'.
        initial_alpha: float. Initial step size.
        termination_conditions: list of TerminationCondition. Conditions to stop the optimization.
        tol: float. Tolerance to stop the optimization.
        max_iter: int. Maximum number of iterations.

    Returns:
        x_next: array-like. Solution found.
        steps: list of array-like. Steps taken by the algorithm.
        num_steps: int. Number of steps taken.
    """
    _validate_inputs(func, tol, max_iter, step_size)

    x_i = np.array(x0)
    gradient = grad(func)

    steps = [x_i.copy()]  # Store steps for plotting
    i = 1
    while True:
        # Direction d_i, step size alpha, and noise
        d_i = -gradient(x_i) / np.linalg.norm(gradient(x_i))
        alpha = choose_step_size(func, step_size, initial_alpha, x_i, d_i, i)
        noise = np.random.normal(0, 1 / i**3, size=x_i.shape)
        x_next = x_i - alpha * d_i + noise  # Update formula with noise

        steps.append(x_next.copy())

        if check_termination(
            termination_conditions, i, x_i, d_i, x_next, tol, max_iter
        ):
            # Log experiment conditions and results to a file
            termination_conditions_str = ", ".join(
                [tc.value for tc in termination_conditions]
            )
            step_strategy = step_size if isinstance(step_size, str) else step_size.value

            logger.info(
                f"Noisy gradient descent \t {x_next} \t {x0} \t {step_strategy} \t {termination_conditions_str} \t {tol} \t {i}"
            )

            return x_next, steps, i

        # Update for next iteration
        i += 1
        x_i = x_next


def momentum_gradient_descent():
    raise NotImplementedError


def conjugate_gradient(
    f: Callable, x0: ArrayLike, tol: float = 1e-7, max_iter: int = 1000
):
    """Conjugate Gradient method to minimize a function using the Fletcher-Reeves formula. 

    Args:
        f: The objective function to minimize.
        grad_f: The gradient of the function.
        x0: Initial guess for the variables.
        tol: Tolerance for convergence.
        max_iter: Maximum number of iterations.

    Returns:
        x: The optimal solution found.
        num_iter: The number of iterations performed.
    """
    x_i = np.array(x0, dtype=float)
    steps = [x_i.copy()]

    # Compute the initial search direction
    grad_f = grad(f)
    g_i = grad_f(x_i)
    d_i = -g_i
    g_prev = g_i

    i = 0
    while np.linalg.norm(g_i) > tol and i < max_iter:
        # Perform a line search to find the step size alpha
        alpha = backtracking_line_search(f, x_i, d_i)

        # Update the current point
        x_i = x_i + alpha * d_i
        steps.append(x_i.copy())

        # Update beta using the Fletcher-Reeves formula
        # beta_i = (g_i^T g_i) / (g_{i-1}^T g_{i-1})
        g_i = grad_f(x_i)
        beta_i = np.dot(g_i, g_i) / np.dot(g_prev, g_prev)

        # Update the search direction
        d_i = -g_i + beta_i * d_i

        # Update g_prev and next iteration
        g_prev = g_i
        i += 1

    return x_i, steps, i


def newton_method_minimize(func, x0, tol=1e-7, max_iter=100):
    """Minimize a function using Newton-Raphson method."""
    x_prev = np.array(x0)
    gradient = grad(func)
    hessian = hess(func)

    steps = [x_prev.copy()]  # Store steps for plotting

    for i in range(max_iter):
        grad_x_prev = gradient(x_prev)
        hess_x_prev = hessian(x_prev)

        if np.linalg.det(hess_x_prev) == 0:
            raise ValueError("Hessian is singular. No solution found.")

        x_next = x_prev - np.linalg.inv(hess_x_prev).dot(grad_x_prev)
        steps.append(x_prev.copy())

        # Terminate if the norm of the gradient is less than the tolerance
        if np.linalg.norm(x_next - x_prev) < tol:
            steps.append(x_next.copy())
            return x_next, steps, i  # Return the steps taken to the minimum

        x_prev = x_next

    return x_next, steps, i  # Return the steps even if not converged


def plot_steps(
    func: Callable,
    steps: list[ArrayLike],
    title: str = None,
    x_range: tuple[float, float] = (0, 3),
    y_range: tuple[float, float] = (0, np.pi / 2),
    ax: Axes = None,
):
    """Plot the steps taken by the optimization algorithm on a contour plot of the function.

    Args:
        func: callable. Function to be optimized.
        steps: list of array-like. Steps taken by the optimization algorithm.
        title: str. Title of the plot.
        x_range: tuple. Range of x values for the contour plot.
        y_range: tuple. Range of y values for the contour plot.
        ax: matplotlib Axes. Axes to plot the steps.
    """
    if not callable(func):
        raise ValueError("func must be a callable function")
    if len(steps[0]) != 2:
        raise ValueError("func must be a 2D function")

    steps = np.array(steps)

    # Create a meshgrid for the contour plot
    x = np.linspace(*x_range, 100)
    y = np.linspace(*y_range, 100)
    X, Y = np.meshgrid(x, y)
    Z = func([X, Y])

    if ax is None:
        _, ax = plt.subplots()

    contour_plot = ax.contour(X, Y, Z, levels=20, cmap="viridis")
    ax.clabel(contour_plot, contour_plot.levels, inline=True, fontsize=6)
    plt.colorbar(contour_plot)

    # Plot steps, start, and end points
    ax.plot(steps[:, 0], steps[:, 1], "ro-")
    ax.plot(
        steps[0, 0],
        steps[0, 1],
        "bo",
        label=f"Start ({steps[0, 0]:.2f}, {steps[0, 1]:.2f})",
    )
    ax.plot(
        steps[-1, 0],
        steps[-1, 1],
        "go",
        label=f"End ({steps[-1, 0]:.2f}, {steps[-1, 1]:.2f})",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Minimize func {func.__name__}")


########## EXAMPLE ##########
def negative_f(a_theta: ArrayLike) -> float:
    """Function to be optimized.

    Args:
        a_theta: array-like. Input values [a, theta].

    Returns:
        float. Area value.
    """
    a, theta = a_theta
    return a * np.sin(theta) * (-3 + a * (1 + np.sin(theta)) - 0.5 * a * np.cos(theta))


# Initial guess
x01 = [1.5, 0]
x02 = [1, np.pi / 4]

# Run the gradient descent algorithm
optimal_a1, steps1, num_steps1 = gradient_descent(
    negative_f, x01, step_size=StepSize.INEXACT
)
optimal_a2, steps2, num_steps2 = gradient_descent(
    negative_f, x02, step_size=StepSize.INEXACT
)

# Plot the steps taken by the algorithm
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plt.suptitle("Gradient Descent Steps for Area Function")
plot_steps(
    negative_f,
    steps1,
    ax=ax1,
    title=f"Backtracking Line Search, converged in {num_steps1} steps",
)
plot_steps(
    negative_f,
    steps2,
    ax=ax2,
    title=f"Backtracking Line Search, converged in {num_steps2} steps",
)
