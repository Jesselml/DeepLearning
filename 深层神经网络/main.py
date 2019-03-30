from deep_nn import DeepNN
import numpy as np

deep_nn = DeepNN()

#1 test initialize_parameters
initial_parameters = deep_nn.initialize_parameters()
# print ("W1",initial_parameters["W1"])
# print ("b1",initial_parameters["b1"])
# print ("W2",initial_parameters["W2"])
# print ("b2",initial_parameters["b2"])

#2 test forward_propagation
X = np.array([1,1,2,1,1,2]).reshape(2,3)
caches = deep_nn.forward_propagation(X,initial_parameters)
print ("All Z and A will be used in backward_propagation:\n",caches)

#3 test the compute cost
Y = np.array([1,0,1]).reshape(1,3)
A_L = caches["A"+str(len(deep_nn.layer_dims)-1)]
cost = deep_nn.compute_cost(A_L,Y)
print ("current cost:",cost)

#4 test backward_propagation
gradients = deep_nn.backward_propagation(Y,caches,initial_parameters)
print ("gradients is:\n",gradients)