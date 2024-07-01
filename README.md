# TrIM: Transformed Iterative Mondrian Forests for Gradient-based Dimension Reduction and High-Dimensional Regression

Ricardo Baptista, Eliza O’Reilly, and Yangxinyu Xie

## Introduction
This is the official implementation of the TrIM algorithm, as described in the paper: [TrIM: Transformed Iterative Mondrian Forests for Gradient-based Dimension Reduction and High-Dimensional Regression]().

## Requirements
- Python 3.11
- JAX

To install JAX, please follow the instructions on the [JAX website](https://github.com/google/jax?tab=readme-ov-file#installation)

To install the remaining required packages, run:
```setup
pip install -r requirements.txt
```

## Implementations of TrIM
The TrIM algorithm is implemented in the `src/Mondrian_RF` folder. Part of the code is based on the [Mondrian Forests](https://github.com/matejbalog/mondrian-kernel) implementation by Matej Balog.

## Experiments
The experiments in the paper can be reproduced by running the following scripts:
- `src/Simulations.ipynb`: Simulation experiments
- `src/Ebola.ipynb`: Ebola experiments
- `src/eval.py`: Real data experiments on machine learning datasets