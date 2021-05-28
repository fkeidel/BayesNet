# BayesNet
This is a simple library for inference in [Bayesian Networks](https://en.wikipedia.org/wiki/Bayesian_network) in C++. It implements some basic algorithms for working with Probabilistic Graphical Models (PGM). It is based on the online Stanford course ["Probabilistic Graphical Models"](https://www.coursera.org/specializations/probabilistic-graphical-models) on Coursera. There are no dependencies to other libraries, except GoogleTest.

## Features
### Probability calculation with "factors"
- factor class (conditional probability table)
- factor product/sum
- (max) marginalization
- joint probability
### Inference algorithms
 - [Variable Elimination algorithm (VE)](https://en.wikipedia.org/wiki/Variable_elimination)
 - [Clique Tree algorithm (Junction Tree)](https://en.wikipedia.org/wiki/Junction_tree_algorithm)
    - cf. [Junction Tree - Stanford CS228](https://ermongroup.github.io/cs228-notes/inference/jt/) 
    - compute marginals
    - compute MAP assignements (decoding)
 - [Dynamic Bayes Nets (DBN)](https://en.wikipedia.org/wiki/Dynamic_Bayesian_network) - [Hidden Markov Models (HMM)](https://en.wikipedia.org/wiki/Hidden_Markov_model)
    - cf. [Dynamic Bayesian Networks: Representation, Inference and Learning (Murphy)](https://www.cs.ubc.ca/~murphyk/Thesis/thesis.html)

### Decision algorithms
- [Influence Diagram](https://en.wikipedia.org/wiki/Influence_diagram) (currently only one decision and one utility factor)

## Benefit
- learn how to do exact inference and solve decision problems
- efficient exact inference with the Clique Tree algorithm (Junction Tree)
- online inference with Dynamical Bayesian Networks (DBN)

## Limitations
- currently only discrete variables are supported
- currently only simple influence diagrams possible (one decision, one utility factor)

## Tutorial
There is a [tutorial](doc/tutorial.ipynb) explaining factor calculation and showing how to run inference on Bayesian Networks.

## Examples
- [Disease-Test](examples/disease_test/disease_test.md) (Bayes Theorem, factor calculation, variable elimination)
- [Water Sprinkler](examples/water_sprinkler/water_sprinkler.md) (variable elimination, clique tree)
- [Simple Bayesian Traffic Jam Detector](examples/traffic_jam/traffic_jam.ipynb) (sensor fusion, Hidden Markov Model, Dynamic Bayesian Network)
- [Markov Random Field grid sampling](examples/grid_sampling/grid_sampling.ipynb) (Markov Random Field, Grid, Sampling)

## Install, build, run
see [How to install, build and run](install-build-run.md)

## License
BayesNet is free software. It is released under the [BSD Zero Clause License](LICENSE).

## Recommended reading/videos
- [Book: Machine Learning: a Probabilistic Perspective (Murphy)](https://probml.github.io/pml-book/book0.html)
- [Thesis: Dynamic Bayesian Networks: Representation, Inference and Learning (Murphy)](https://www.cs.ubc.ca/~murphyk/Thesis/thesis.html)
- [Book: Probabilistic Graphical Models: Principles and Techniques (Koller, Friedman)](https://mitpress.mit.edu/books/probabilistic-graphical-models)
- [Video: Graphical Models - Christopher Bishop - Machine Learning Summer School 2013](https://youtu.be/ju1Grt2hdko)
- [Lecture Notes: Standford CS 228 - Probabilistic Graphical Models](https://ermongroup.github.io/cs228-notes/)

## Tools
- [Open Markov](http://www.openmarkov.org/): Open-source software tool for Probabilistic Graphical Models in Java

---
<br>

![](examples/water_sprinkler/water_sprinkler.svg)
