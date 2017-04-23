from neuralnet.neuralnet import NeuralNet
import numpy as np

from util.numpyextensions import NumpyExtensions
np.random.seed(1)
X = np.array([[0, 1], [0, 0], [1, 0], [1, 1]])
y = np.array([[1, 0, 1, 0]]).T

CORRECT = 0
FAILED = 0

def run_neural_net(iter, X, y):
    print("Running iteration number",iter)
    nn = NeuralNet(2, 1, 3)
    nn.train(X, y, learning_rate=4)
    print(nn.cost(X,y))
    return nn.cost(X,y) < 1e-3

for i in range(0, 1):
    if run_neural_net(i, X, y):
        CORRECT += 1
    else:
        FAILED += 1


print("Correct:", CORRECT, "Failed:", FAILED)