from neuralnet.neuralnet import NeuralNet
import numpy as np

from util.numpyextensions import NumpyExtensions

X = np.array([[0, 1], [0, 0], [1, 0], [1, 1]])
y = np.array([[1, 0, 1, 0]]).T

# nn = NeuralNet(2, hd_layers= 1, hd_layers_size=2)
nn = NeuralNet(2, 1, 8)

print(nn.thetas)
#print(NumpyExtensions.reshape_for_neuralnet(NumpyExtensions.flatten(nn.thetas), nn))

#nn.check_gradients(X, y, True)
#print("NUmerical", nn.numerical_gradients(X, y))
#print("Algorithmical", nn.gradients(X, y))
#print(nn.cost(X, y))

#print(nn.thetas, thetas)
#nn.thetas = [np.array([[-30], [20], [20]])]
#print(nn.forward_propagate(X))
#print(nn.gradients(X, y))
nn.train(X, y)
#print(nn.thetas)
#print(nn.cost(X, y))


print(np.around(nn.predict(X), decimals = 2))