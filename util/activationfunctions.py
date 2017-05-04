import numpy as np
import scipy.special


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
            return ActivationFunctions.sigmoid(z) * (1-ActivationFunctions.sigmoid(z))
        # Catch boundaries, as precision loss will lead to the result being 0 or 1.
        # However, this leads to further errors when calculating the cost.
        z[z < -709] = -709
        z[z > 19] = 19
        s = 1.0 / (1.0 + np.exp(-z))
        return s