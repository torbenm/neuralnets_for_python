from neuralnet.calculation.base import CalculationBase
from util.numpyextensions import NumpyExtensions as npe
import numpy as np

from util.activationfunctions import ActivationFunctions


class NeuralNetClassification(CalculationBase):

    activate = ActivationFunctions.sigmoid

    def __init__(self, neuralnet):
        CalculationBase.__init__(self, neuralnet)

    def _forwardpropagate(self, thetas, X):
        # Load vars from neural net
        nl = self.neuralnet.num_layers()

        # Init variables
        a = [None]*(nl+1) # Activation values, equal to z with applied activation method
        z = [None]*(nl+1) # z values
        a[0] = X # Starting with X

        # Iterate over values to calculate a,z
        for i in range(0, nl):
            a[i] = npe.add_ones(a[i])
            z[i+1] = np.dot(a[i], thetas[i])
            a[i+1] = NeuralNetClassification.activate(z[i+1]) # z for next level, i+1

        # Return last z values, a's, and z's
        return a[nl],a,z

    def _cost(self, thetas, X, y):
        # Do forward propagation
        h,_,_ = self._forwardpropagate(thetas, X)

        # Calculate cost
        total_cost = (y * np.log(h)) + (1 - y)*np.log(1 - h)

        # Get the average cost
        return (-1/(X.shape[0])) * np.sum(np.sum(total_cost))

    def _gradients(self, thetas, X, y):
        # Load vars from neural net
        nl = self.neuralnet.num_layers()

        # Init variables
        DELTA = [None]*(nl)
        GRADIENTS = [None]*nl
        delta = [None]*(nl+1)
        m = X.shape[0]

        # Do forward propagation, but here we don't need the result but the built vars
        _,a,z = self._forwardpropagate(thetas,X)


        #print(a[nl].shape, y.shape)
        delta[nl] = npe.add_ones(a[nl] - y).T

        # Calculate deltas
        for i in reversed(range(1, nl)):
            delta[i] = np.dot(thetas[i], delta[i+1][1:]) \
                       * NeuralNetClassification.activate(a[i].T, True)
        # Calculate DELTAS
        for i in range(0, nl):
            DELTA[i] = np.dot(delta[i+1][1:], a[i]).T

        # Calculate GRADIENTS
        for i in range(0, nl):
            GRADIENTS[i] = (1/m) * DELTA[i]

        return GRADIENTS
