# DEEP DECLARATIVE NODES
# Defines the interface for data processing nodes and declarative nodes. The implementation here is kept simple
# with inputs and outputs assumed to be vectors. There is no distinction between data and parameters and no
# concept of batches. For using deep declarative nodes in a network for end-to-end learning see code in the
# `ddn.pytorch` package.
#
# Stephen Gould <stephen.gould@anu.edu.au>
# Dylan Campbell <dylan.campbell@anu.edu.au>
#

import autograd.numpy as np
import scipy as sci
from autograd import grad, jacobian
import warnings

class AbstractNode:
    """
    Minimal interface for generic data processing node that produces an output vector given an input vector.
    """

    def __init__(self, n=1, m=1):
        """
        Create a node
        :param n: dimensionality of the input (parameters)
        :param m: dimensionality of the output (optimization solution)
        """
        assert (n > 0) and (m > 0)
        self.dim_x = n # dimensionality of input variable
        self.dim_y = m # dimensionality of output variable

    def solve(self, x):
        """Computes the output of the node given the input. The second returned object provides context
        for computing the gradient if necessary. Otherwise it's None."""
        raise NotImplementedError()
        return None, None

    def gradient(self, x, y=None, ctx=None):
        """Computes the gradient of the node for given input x and, optional, output y and context cxt.
        If y or ctx is not provided then they are recomputed from x as needed."""
        raise NotImplementedError()
        return None


class AbstractDeclarativeNode(AbstractNode):
    """
    A general deep declarative node defined by an unconstrained parameterized optimization problems of the form
        minimize (over y) f(x, y)
    where x is given (as a vector) and f is a scalar-valued function. Derived classes must implement the `objective`
    and `solve` functions.
    """

    eps = 1.0e-6 # tolerance for checking that optimality conditions are satisfied

    def __init__(self, n=1, m=1):
        """
        Creates an declarative node with optimization problem implied by the objecive function. Initializes the
        partial derivatives of the objective function for use in computing gradients.
        """
        super().__init__(n, m)

        # partial derivatives of objective
        self.fY = grad(self.objective, 1)
        self.fYY = jacobian(self.fY, 1)
        self.fXY = jacobian(self.fY, 0)

    def objective(self, x, y):
        """Evaluates the objective function on a given input-output pair."""
        warnings.warn("objective function not implemented.")
        return 0.0

    def solve(self, x):
        """
        Solves the optimization problem
            y in argmin_u f(x, u)
        and returns two outputs. The first is the optimal solution y and the second contains the context for
        computing the gradient, such as the largrange multipliers in the case of a constrained problem, or None
        if no context is available/needed.
        """
        raise NotImplementedError()
        return None, None

    def gradient(self, x, y=None, ctx=None):
        """
        Computes the gradient of the output (problem solution) with respect to the problem
        parameters. The returned gradient is an ndarray of size (self.dim_y, self.dim_x). In
        the case of 1-dimensional parameters the gradient is a vector of size (self.dim_y,).
        Can be overridden by the derived class to provide a more efficient implementation.
        """

        # compute optimal value if not already done so
        if y is None:
            y, ctx = self.solve(x)
        assert self._check_optimality_cond(x, y)

        return -1.0 * sci.linalg.solve(self.fYY(x, y), self.fXY(x, y), assume_a='pos')

    def _check_optimality_cond(self, x, y, ctx=None):
        """Checks that the problem's first-order optimality condition is satisfied."""
        return (abs(self.fY(x, y)) <= self.eps).all()