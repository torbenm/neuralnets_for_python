from abc import abstractmethod

from util.numpyextensions import NumpyExtensions as npe

class CalculationBase(object):

    neuralnet = None

    def __init__(self, neuralnet):
        self.neuralnet = neuralnet

    # FORWARD PROPAGATION

    @abstractmethod
    def _forwardpropagate(self, theta, X):
        pass

    def forwardpropagate(self, X):
        r,_,_ = self._forwardpropagate(self.neuralnet.thetas, X)
        return r


    def _flat_forwardpropagate(self, flat_thetas, X, *args):

        thetas = npe.reshape_for_neuralnet(flat_thetas,
                                  self.neuralnet)

        r,_,_ = self._forwardpropagate(thetas, self.neuralnet.num_layers(), X)
        return npe.flatten(r)

    # COST

    @abstractmethod
    def _cost(self, theta, X, y):
        pass

    def _flat_cost(self, flat_thetas, X, y, *args):
        thetas = npe.reshape_for_neuralnet(flat_thetas, self.neuralnet)
        return self._cost(thetas, X, y)

    def cost(self, X, y):
        return self._cost(self.neuralnet.thetas, X, y)

    # REGULARIZED COST

    @abstractmethod
    def _regularized_cost(self, theta, X, y, l):
        pass

    def _flat_regularized_cost(self, flat_thetas, X, y, l, *args):
        thetas = npe.reshape_for_neuralnet(flat_thetas, self.neuralnet)
        return self._regularized_cost(thetas, X, y, l)

    def regularized_cost(self, X, y, l):
        return self._regularized_cost(self.neuralnet.thetas, X, y, l)

    # GRADIENTS

    @abstractmethod
    def _gradients(self, theta, X, y):
        pass

    def _flat_gradients(self, flat_thetas, X, y, *args):
        thetas = npe.reshape_for_neuralnet(flat_thetas, self.neuralnet)
        gradients = self._gradients(thetas, X, y)
        f = npe.flatten(gradients)
        return f

    def gradients(self, X, y):
        return self._gradients(self.neuralnet.thetas, X, y)

    # REGULARIZED GRADIENTS
    
    @abstractmethod
    def _regularized_gradients(self, theta, X, y, l):
        pass

    def _flat_regularized_gradients(self, flat_thetas, X, y, l, *args):
        thetas = npe.reshape_for_neuralnet(flat_thetas, self.neuralnet)
        gradients = self._regularized_gradients(thetas, X, y, l)
        f = npe.flatten(gradients)
        return f

    def regularized_gradients(self, X, y, l):
        return self._regularized_gradients(self.neuralnet.thetas, X, y, l)
