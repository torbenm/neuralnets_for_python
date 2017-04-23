import numpy as np
from scipy.optimize import fmin_cg

from neuralnet.calculation.classification import NeuralNetClassification
from optimization.gradientdescent import gradient_descent
from util.numpyextensions import NumpyExtensions as npe


class NeuralNet(object):

    thetas = []
    input_variables = 0
    output_variables = 1
    hidden_layers = 0
    hidden_units = 0
    epsilon = 0.12

    def __init__(self, input_variables, hd_layers = 0, hidden_units = 0, output_variables = 1, epsilon = 0.12):
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.hidden_layers = hd_layers
        self.hidden_units = hidden_units
        self.epsilon = epsilon
        self.rand_init()

    def rand_init(self):
        self.thetas = [None]*self.num_layers()
        for i in range(0, self.num_layers()):
            input = self.input_variables if i == 0 else self.hidden_units
            output = self.output_variables if i == self.hidden_layers else self.hidden_units
            self.thetas[i] = npe.random_matrix(input, output, epsilon=self.epsilon)

    def num_layers(self):
        return self.hidden_layers + 1

    def predict(self, X):
        return NeuralNetClassification(self).forwardpropagate(X)

    def cost(self, X, y):
        return NeuralNetClassification(self).cost(X, y)

    def gradients(self, X, y):
        return NeuralNetClassification(self).gradients(X, y)

    def numerical_gradients(self, X, y, epsilon = 1e-4):
        flat_thetas = npe.flatten(self.thetas)
        numgrad = np.zeros(flat_thetas.shape)
        perturb = np.zeros(flat_thetas.shape)
        for i in range(0, flat_thetas.size):
            perturb[i] = epsilon
            loss1 = self.get_algorithms()._flat_cost(flat_thetas - perturb, X, y)
            loss2 = self.get_algorithms()._flat_cost(flat_thetas + perturb, X, y)
            numgrad[i] = (loss2 - loss1)/ (2*epsilon)
            perturb[i] = 0
        return numgrad

    def p_numerical_gradients(self, thetas, X, y, epsilon = 1e-4):
        flat_thetas = npe.flatten(thetas)
        numgrad = np.zeros(flat_thetas.shape)
        perturb = np.zeros(flat_thetas.shape)
        for i in range(0, flat_thetas.size):
            perturb[i] = epsilon
            loss1 = self.get_algorithms()._flat_cost(flat_thetas - perturb, X, y, False)
            loss2 = self.get_algorithms()._flat_cost(flat_thetas + perturb, X, y, False)
            numgrad[i] = (loss2 - loss1)/ (2*epsilon)
            perturb[i] = 0
        return numgrad

    def check_gradients(self, X, y, do_print = False, imprecision = 1e-4, epsilon = 1e-8):
        flat_thetas = npe.flatten(self.thetas)
        algorithmic = self.get_algorithms()._flat_gradients(flat_thetas, X, y)
        numerical = self.numerical_gradients(X, y, epsilon)
        all_match = True
        for i in range(0, flat_thetas.size):
            is_match = abs(algorithmic[i] - numerical[i]) < imprecision
            all_match = all_match and is_match
            if(do_print):
                print(i, numerical[i], algorithmic[i], is_match)
        if(do_print and all_match):
            print("All gradients are matching")
        elif(do_print and not all_match):
            print("Not all gradients are matching")
        return all_match

    def p_check_gradients(self, thetas, X, y, do_print = False, imprecision = 1e-4, epsilon = 1e-8):
        flat_thetas = npe.flatten(thetas)
        algorithmic = self.get_algorithms()._flat_gradients(flat_thetas, X, y)
        numerical = self.p_numerical_gradients(thetas, X, y, epsilon)
        all_match = True
        for i in range(0, flat_thetas.size):
            is_match = abs(algorithmic[i] - numerical[i]) < imprecision
            all_match = all_match and is_match
            if(do_print):
                print(i, numerical[i], algorithmic[i], is_match)
        if(do_print and all_match):
            print("All gradients are matching")
        elif(do_print and not all_match):
            print("Not all gradients are matching")
        return all_match

    def get_algorithms(self):
        return NeuralNetClassification(self)

    def train(self, X, y, learning_rate = 0.1, cost_threshold=1e-6, diff_threshold=1e-16, max_iter=10000, min_iter=0):
        nclass = NeuralNetClassification(self)
        # self.thetas = npe.reshape_for_neuralnet(fmin_cg(
        #         f=nclass._flat_cost,
        #         x0=npe.flatten(self.thetas),
        #         fprime=nclass._flat_gradients,
        #         args=(X, y),
        #         epsilon=epsilon,
        #         gtol=gtol,
        #         disp=False,
        #         maxiter=maxiter), self)
        self.thetas = npe.reshape_for_neuralnet(gradient_descent(
            start_thetas=npe.flatten(self.thetas),
            cost_func=nclass._flat_cost,
            gradient_func=nclass._flat_gradients,
            args=(X,y),
            learning_rate=learning_rate,
            min_iter=min_iter,
            max_iter=max_iter,
            cost_threshold=cost_threshold,
            diff_threshold=diff_threshold
        ), self)