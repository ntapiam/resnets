# Companion code for "Stability of deep ResNets via discrete rough paths"

This repository contains the companion code for the numerical experiments
presented in the paper "Stability of deep ResNets via discrete rough paths".

The numerical expermints consist in two parts:
1. Training of a Residual Network consisting of 512 residual blocks on the
   MNIST data set for a total of 100 epochs. The weights are then saved to
   disk.
2. Computing the $p$-variation (for $p\in[1,3]$) of those weights using the
   Euclidean norm for the vector norm and the Frobenius norm for the matrix
   norm. The results of this step are then plotted and saved to disk.

Step 1. is performed by the code contained in `resnet.py`. This file uses
the PyTorch Machine Learning framework [^1]. Step 2. is performed by the code
contained in `iss.py`. This file implements the iterated-sums signature in
PyTorch, and uses a port of T. Lyons', A. Korepanov's and P. Zorin-Kranich's
`p-var` C++ library [^2] to the Rust programming language, and exported as a
Python module. This port was written by the authors of the current package.

All the code runs inside a `pipenv` environment pinning all Python modules
to the exact version used during development.
In order to build the Rust extensions, a working Rust installation is
needed (see [here](https://rust-lang.org/tools/install) for installation
instructions).

## How to run the code

First, setup the `pipenv` environment by running `pipenv sync` to install all
the dependencies.
Then build the `p_var` extension by compiling it with `pipenv run maturin
develop --release`. This will compile the Rust extension and install it inside
the `pipenv` virtual environment.

Train the ResNet by running `pipenv run python3 resnet.py`. This will produce
a file on disk with the `.pth` extension containing the trained weights. Note
that some configuration of the training setup could be needed before the code
fully trains in your particular machine (e.g. type and number of accelerators,
etc), although they should be automatically discovered in most cases.

Finally, compute the p-variations by running `pipenv run python3 iss.py`.
This will produce a numpy file with extension `.pth` containing the numerical
data and a PDF figures showing the evolution and $p$-variation norm of the
weights.

[^1]: https://pytorch.org/
[^2]: https://github.com/khumarahn/p-var.git
