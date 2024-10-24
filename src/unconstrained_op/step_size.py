"""Module for line search methods to find optimal step size, including exact and inexact line search.

1. bisection_method: Bisection method to find the minimum of a function within an interval.
2. backtracking_line_search: Inexact line search algorithm to find the optimal step size.
"""

from collections.abc import Callable

import autograd.numpy as np
from autograd import grad
from numpy.typing import ArrayLike


def bisection_method(
    f: Callable, a: float, b: float, eps: float = 1e-8
) -> tuple[float, float]:
    """Bisection method to find the minimum of function f within the interval [a, b].

    Inputs:
    - f: The function to minimize.
    - a: Left endpoint of the interval.
    - b: Right endpoint of the interval.
    - eps: The precision level for termination (default is 1e-5).

    Output:
    - The bracketed interval [a, b] containing the local minimum.
    """
    if a >= b:
        a, b = b, a

    grad_f = grad(f)

    # If either f'(a) or f'(b) doesn't meet conditions, expand the interval
    f_prime_a = grad_f(a)
    f_prime_b = grad_f(b)

    if f_prime_a >= 0 or f_prime_b <= 0:
        a, b = expand_interval(f, grad_f, a, b)

    while abs(b - a) > eps:
        m = (a + b) / 2
        f_prime_m = grad_f(m)

        if f_prime_m > 0:
            b = m
        else:
            a = m

    return float(a), float(b)


def expand_interval(
    f: Callable, grad_f: Callable, a: float, b: float, w_factor: float = 1.0
) -> tuple[float, float]:
    """Expands the interval [a, b] until f'(a) < 0 and f'(b) > 0 are satisfied.

    Inputs:
    - f: The function to minimize.
    - grad_f: The gradient of the function.
    - a: Left endpoint of the interval.
    - b: Right endpoint of the interval.
    - w_factor: Factor for expanding the interval (default is 1.0).

    Output:
    - The new interval [a, b].
    """
    while True:
        w = abs((b - a) / 2)

        f_prime_a = grad_f(a)
        f_prime_b = grad_f(b)

        if f_prime_a < 0 and f_prime_b > 0:
            return a, b

        a = a - w_factor * w
        b = b + w_factor * w

    return a, b


def backtracking_line_search(
    f: Callable,
    x_i: ArrayLike,
    d: ArrayLike | None = None,
    alpha: float = 1.0,
    p: float = 0.5,
    c: float = 1e-4,
) -> float:
    """
    Backtracking line search algorithm to find the optimal step size alpha.

    Inputs:
    - f: Objective function
    - x_i: Current point (numpy array)
    - d: Descent direction (numpy array)
    - alpha: Initial step size (default is 1.0)
    - p: Scaling factor to reduce alpha (default is 0.5)
    - c: Constant for the Armijo condition (default is 1e-4)

    Output:
    - alpha: Step size that satisfies the Armijo condition
    """
    grad_f = grad(f)
    if d is None:
        d = -grad_f(x_i) / np.linalg.norm(grad_f(x_i))  # Steepest descent direction

    while f(x_i + alpha * d) > f(x_i) + c * alpha * np.dot(grad_f(x_i), d):
        alpha *= p  # Reduce alpha by factor p

    return alpha


def f(x):
    return x[0] ** 2 + x[1] ** 2


x_i = np.array([2, 1])

optimal_alpha = backtracking_line_search(f, x_i)
print(f"Optimal alpha: {optimal_alpha}")
