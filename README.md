# InferDynamic

Infer a dynamic system

### Overview

- We presented three methods for black-box identification of switched nonlinear dynamical systems from trajectory data, and proposed a way to evaluate an inferred model by comparison to the original model. Using this evaluation, we tested the three methods for robustness and compared them on five classes of 20 examples in total. 

### Installation & Usage

#### Prerequisite

- Python 3.7 with libraries SciPy, scikit-learn and LIBSVM


#### Installation

- Just download. We have tested on an 1.8GHz Intel Core-i7 8550U processor with 8GB RAM running 64-bit Windows

#### Usage

- Run run_tests.py

#### Output

- Each row of results represents an experiment on a system and these components are respectively 'the variant ID' , 'the example ID', 'number of initial points', 'time step size', 'interval of simulation', 'the absolute error tolerance in Method 2', 'average relative distances using Method 1, Method 2, and Method 3' and 'wall-clock inference time in seconds using Method 1, Method 2, and Method 3'.
