import numpy as np


class ActivationFunctions(object):
    @staticmethod
    def sigmoid(z, derivation: bool = False):
        """
        Calculates the sigmoid function of a value z.
        If derivaton is set to true, the derivation of the sigmoid function is used.
        Read more about the sigmoid function here: https://en.wikipedia.org/wiki/Sigmoid_function

        :param z: The value to calculate the sigmoid with.
                  If z is a vector or matrix, the sigmoid function will be calculated element-wise
        :param derivation: If True, the derivation of the sigmoid function is used. Default is false.
        :return: The result of the sigmoid function. It has the same shape as z
        """
        if derivation:
            return z * (1 - z)
        return 1 / (1 + np.exp(-z))
