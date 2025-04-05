# A Beginner's Guide to Variational Inference

## Description

This tutorial is **for data scientists, statisticians, and machine learning practitioners who are comfortable with Python and basics of probability**.

We’ll break down the mechanics of VI and its application in PyMC in an approachable way, starting with intuitive explanations and building up to practical examples.

Participants will learn how to apply ADVI and Pathfinder in PyMC and evaluate their results against MCMC, gaining insights into when and why to choose VI.

### Takeaways

Participants will leave understanding:

- The fundamentals of VI and how it differs from MCMC.
- How to implement ADVI and Pathfinder in PyMC.
- Practical considerations when selecting and evaluating inference methods.

### Background Knowledge Required

- Basic understanding of probability and Bayesian inference.
- Familiarity with Python. Prior PyMC experience is helpful but not required.

### Materials Distribution

All materials, including notebooks and datasets, will be available on GitHub.

### Setting up the Environment

If using Anaconda/Miniforge:
The repository contains an `environment.yml` file with all required packages. Run:

    mamba env create

if you are using Miniforge, or if you installed Anaconda, you can use:

    conda env create

from the main course directory (use `conda` instead of `mamba` if you installed Anaconda). Then activate the environment:

    mamba activate pymc_vi_course
    # or
    conda activate pymc_vi_course

If using Pixi:
The repository contains a `pixi.toml` file. From the main course directory, simply run:

    pixi install
    pixi shell

Then, you can start **JupyterLab** to access the materials:

    jupyter lab

For those who like to work in VS Code, you can also run Jupyter notebooks from within VS Code. To do this, you will need to install the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter). Once this is installed, you can open the notebooks in the `notebooks` subdirectory and run them interactively.

## Outline

1. **Introduction: Why Variational Inference?** (10 min)
- The limitations of MCMC for large datasets.
- Overview of VI: How it works and why it’s faster.

2. **Variational Inference Basics** (20 min)
- Key concepts: Evidence Lower Bound (ELBO), optimization, and approximation families.
- Intuitive explanation of ADVI and Pathfinder.

3. **Implementing VI with PyMC** (15 min)
- Step-by-step walkthrough of VI with a linear model.
- Comparing ADVI, Pathfinder, and MCMC.

4. **Evaluating VI Approximations** (10 min)
- How to measure the quality of VI approximations (ELBO, simulation-based calibration, etc.).
- Practical trade-offs between speed and accuracy.

5. **Scaling Up: Complex Models and Real-World Applications** (25 min)
- Applying VI to hierarchical and large-scale models.
- Tips for debugging and optimizing VI workflows.

6. **Open Discussion and Q&A** (10 min)
- Address audience-specific use cases and questions.
