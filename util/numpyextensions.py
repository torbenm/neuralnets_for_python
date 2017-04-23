import numpy as np


class NumpyExtensions(object):
    @staticmethod
    def random_matrix(input_size, output_size, epsilon=0.12, input_bias=1):
        """
        Creates a random matrix that can be used as theta initialisation for neural nets.

        :param epsilon: How far the values will diverge from each other. 0.12 by default.
        :param input_size: The number of input units.
        :param output_size: The number of output units
        :param input_bias: Whether there is an input bias or not depends on the number of this value.
                           1 by default.
        :return: A (input_size+input_bias) x output_size matrix
        """
        return np.random.random((input_size + input_bias, output_size)) * 2 * epsilon - epsilon

    @staticmethod
    def add_ones(X, axis=1):
        """
        Adds ones in rows/columns to the given matrix X.
        :param X: The matrix which should be extended.
        :param axis: The axis to which insert the ones. Default is 1, thus adds a
        column of ones. With 0, it adds a row of ones
        :return: The matrix X, extended by ones.
        """
        return np.insert(X, 0, [1], axis=axis)

    @staticmethod
    def flatten(M) -> np.ndarray:
        """
        Flattens a Matrix M to a single-dimension array.
        This matrix can be up to three dimensional.
        However, at least the second dimension must be of type :class:`numpy.ndarray`

        :param M: The matrix to flatten
        :return: The flattened matrix M
        """
        return np.array([item for a in M for item in a.flatten().tolist()])

    @staticmethod
    def reshape(flat_M, input_size, hidden_layers, hidden_units, output_size):
        """
        Reshapes a flattened matrix flat_M for a neural net of the given parameters.

        :param flat_M: The flat matrix
        :param input_size: The number if input features of the neural net
        :param hidden_layers: The number of hidden layers
        :param hidden_units: The number of hidden units per hidden layer
        :param output_size: The number of output features
        :return: The reshaped matrix
        """
        M = [None] * (hidden_layers + 1)
        for i in range(0, len(M)):
            if i == 0:
                rows = input_size + 1
                start = 0
            else:
                start = (input_size + 1) * hidden_units + (i - 1) * (
                    (hidden_units + 1) * hidden_units)
                rows = hidden_units + 1

            if i == hidden_layers:
                columns = output_size
            else:
                columns = hidden_units
            M[i] = np.asarray(flat_M[start:start + (rows * columns)]).reshape([rows, columns])
        return M

    @staticmethod
    def reshape_for_neuralnet(flat_M, neuralnet):
        """
        Reshapes the given flat matrix with the values for the given neural net.
        To do so, it calls :func:`reshape`

        :param flat_M: The flat matrix
        :param neuralnet: The neural net.
        :return: The reshaped matrix
        """
        return NumpyExtensions.reshape(flat_M,
                                       neuralnet.input_variables,
                                       neuralnet.hidden_layers,
                                       neuralnet.hidden_layer_size,
                                       neuralnet.output_variables)