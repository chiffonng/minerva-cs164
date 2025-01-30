# Minerva CS164 Optimization Methods

This repository contains the code for the course CS164 Optimization Methods at Minerva University (Fall 2024), and some utilities for the course. Code is written in Python and SageMath (optionally).

_Notes_: I'm still taking the course, so some of the code may be incorrect, incomplete, not optimized, or not working. I'll update the code as I progress through the course.

## Curriculum

The course is taught by Prof John Levitt and roughly follows the textbook [Kochenderfer, M. J., & Wheeler, T. A. (2019). *Algorithms for optimization*](https://algorithmsbook.com/optimization/files/optimization.pdf), with several class problems drawn from [Boyd and Vandenberghe (2018) _Convex optimization_](https://web.stanford.edu/~boyd/cvxbook/). The class meets twice a week for 90 minutes each session in a seminar/flipped classroom format, for a total of 25 sessions.

- Session 1: Introduction to optimization

- Session 2: Taylor series and numerical approximation
- Session 3: Quadratic forms
- Session 4: Tests for positive definiteness
- Session 5: Bracketing methods
- Session 6: Introduction to descent (& line search)
- Session 7: Gradient descent
- Session 8: Conjugate gradient
- Session 9: Momentum and noisy gradient descent
- Session 10: Newton's method
- Session 11: Unconstrained optimization review

- Session 12: Equality constraints with Lagrange multipliers
- Session 13: Convexity
- Session 14: KKT conditions
- Session 15: Interpreting the KKT conditions
- Session 16: Linear programming
- Session 17: Duality
- Session 18: Integer programming: branch and bound
- Session 19: Mixed integer programming
- Session 20: Quadratic programming
- Session 21: LMIs 
- Session 22: Semindefinite programming
- Session 23: Newton's method for linear equality constraints
- Session 24: Barrier methods
- Session 25: Review

## Setup

Run the following command in terminal to create the `.venv` virtual environment with the required packages:

```bash
uv venv
source .venv/bin/activate
uv install
```

For Sage setup, refer to [this](https://doc.sagemath.org/html/en/installation/launching.html#setting-up-sagemath-as-a-jupyter-kernel-in-an-existing-jupyter-notebook-or-jupyterlab-installation)

- [ ] TODO: Remove SageMath dependency

## Convert markdown to notebook

```bash
jupytext --to notebook "markdown file.md"
```

## Before commits

```bash
pre-commit install
pre-commit run --all-files
```
