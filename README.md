# Minerva CS164 Optimization Methods

This repository contains the code for the course CS164 Optimization Methods at Minerva University, and some utilities for the course. Code is written in Python and SageMath (optionally). In the course, I learned the math and rewrote some of the algorithms used in [scipy.optimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) for educational purposes.

_Notes_: I'm still taking the course, so some of the code may be incorrect, incomplete, not optimized, or not working. I'll update the code as I progress through the course.

## Setup

Have `conda` installed. Run the following command in terminal to create the `conda` virtual environment with the required packages:

```bash
bash setup.sh
```

To export the `conda` environment to `requirements.txt`:

```bash
pip list --format=freeze > requirements.txt
```

For Sage setup, refer to [this](https://doc.sagemath.org/html/en/installation/launching.html#setting-up-sagemath-as-a-jupyter-kernel-in-an-existing-jupyter-notebook-or-jupyterlab-installation)

- [ ] TODO: Remove SageMath dependency

## Convert markdown to notebook

```bash
jupytext --to notebook "markdown file.md"
```

## Lint and format

```bash
ruff format
ruff check --fix
```
