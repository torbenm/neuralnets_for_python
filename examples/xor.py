from neuralnet.neuralnet import NeuralNet
import numpy as np

# X values and their y values.
X = np.array([[0, 1], [0, 0], [1, 0], [1, 1]])
y = np.array([[1, 0, 1, 0]]).T

# We will build a neuralnet with 2 features, 1 hidden layer with 2 hidden units
neuralnet = NeuralNet(2, 1, 3)

# Let's see what our cost is before training and the predictions
print("Cost before training", neuralnet.cost(X, y))
print("Predictions before training", neuralnet.predict(X))

# Let's train the neural net
print("Training the neural net....")
neuralnet.train(X, y, learning_rate=7)

print("Cost after training", neuralnet.cost(X, y))
print("Predictions after training", np.around(neuralnet.predict(X)))
