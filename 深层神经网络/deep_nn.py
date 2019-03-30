import numpy as np

class DeepNN:
    def __init__(self):
        self.layer_dims = [2,2,1]
        self.L = len(self.layer_dims)-1
        self.parameters = dict()

    def initialize_parameters(self):
        """
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network
        
        Returns:
        parameters -- python dictionary containing parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        """
        np.random.seed(3)
        parameters = {}

        for l in range(1, len(self.layer_dims)):
            parameters['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * 0.01
            parameters['b' + str(l)] = np.zeros((self.layer_dims[l], 1))
            
            assert(parameters['W' + str(l)].shape == (self.layer_dims[l], self.layer_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (self.layer_dims[l], 1))

        return parameters
    
    def relu(self,Z):
        return np.array([x[0] if x[0] >0 else 0 for x in Z.reshape(-1,1)]).reshape(Z.shape)
    
    def sigmoid(self,Z):
        return 1/(1+np.exp(-Z))

    def forward_propagation(self,X,parameters):
        #Get the cache Z_l and A_l, which will feed in backward function
        caches = dict()

        #Calculate the A for hidden layer, g() = RELU
        A_l = X
        caches['A' + str(0)] = X

        for l in range(1,self.L):
            A_previous = A_l
            Z_l = parameters['W' + str(l)].dot(A_previous) + parameters['b' + str(l)]
            A_l = self.relu(Z_l)
            caches['Z' + str(l)] = Z_l
            caches['A' + str(l)] = A_l
        
        #Calculate the y_hat for oytput layer, g() = sigmoid
        Z_L = parameters['W' + str(self.L)].dot(A_l) + parameters['b' + str(self.L)]
        A_L = self.sigmoid(Z_L)
        caches['Z' + str(self.L)] = Z_L
        caches['A' + str(self.L)] = A_L
        assert(caches['A' + str(self.L)].shape == (1,X.shape[1]))

        return caches
    
    def compute_cost(self,AL,Y):
        m = Y.shape[1]
        cost = -1/m * np.sum( Y*np.log(AL)+(1-Y)*(np.log(1-AL)))
        return cost

    def backward_propagation(self,Y,caches,parameters):
        """         
        Returns:
        gradients -- A dictionary with the gradients
        grads["dW" + str(l)] = ...
        grads["db" + str(l)] = ... 
        """
        gradients = dict()
        m = Y.shape[1]

        dA_L = - (np.divide(Y,caches['A' + str(self.L)]) - np.divide(1 - Y, 1 - caches['A' + str(self.L)]))
     
        #Calculate the d for output layer, g() = sigmoid
        dZ_L = dA_L - Y
        dW_L = 1/m * np.dot(dZ_L,caches['A' + str(self.L-1)].T)
        db_L = 1/m * np.sum(dZ_L, axis = 1)
        gradients['dW' + str(self.L)] = dW_L
        gradients['db' + str(self.L)] = db_L

        dA_l_previous = parameters['W' + str(self.L)].T.dot(dZ_L)

        #Calculate the d for hidden layer, g() = RELU
        for l in range(self.L-1,0,-1):
            print ("l",l)
            dA_l = dA_l_previous
            Z_l = caches['Z' + str(l)]
            dZ_l = dA_l * np.array([1 if x[0] >0 else 0 for x in Z_l.reshape(-1,1)]).reshape(Z_l.shape)
            dW_l = 1/m * np.dot(dZ_l,caches['A' + str(l-1)].T)
            db_l = 1/m * np.sum(dZ_l, axis = 1)
            dA_l_previous = parameters['W' + str(l)].T.dot(dZ_l)

            gradients['dW' + str(l)] = dW_l
            gradients['db' + str(l)] = db_l
        
        return gradients
    
    def update_parameters(self,parameters,gradients,learning_rate=1.2):
        for l in range(1,self.L+1):
            parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * gradients["dW" + str(l)]
            parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * gradients["db" + str(l)]
            
        return parameters

    def fit(self,X,Y,n_iterations = 1e5):
        """  
        Arguments:       
        X -- dataset of shape (num of features, number of examples)
        Y -- labels of shape (1, number of examples)
        num_iterations -- Number of iterations in gradient descent loop 
        """
        def gradient_descent(X,Y,n_iterations,initial_parameters,epsilon = 1e-7):
            cur_iterations = 0
            parameters = initial_parameters

            while cur_iterations < n_iterations:
                #1 Forward_propagation to get the essential data for backward_propagation 
                #caches:dict  {"Z1": Z1,"A1": A1,"Z2": Z2,"A2": A2} 
                caches = self.forward_propagation(X,parameters)

                #2 Get the cost for current W and b (int:cost)  
                #last_cost means cost before the descent
                last_cost = self.compute_cost(caches["A"+str(self.L)],Y)

                #3 Backward_propagation to calculate the gradient for W and b
                #gradients:dict   {"dW1":dW1,"dW2":dW2,"db1":db1,"db2":db2}
                gradients = self.backward_propagation(Y,caches,parameters)

                #4 Gradient descent parameter update  Inputs: "parameters, grads". Outputs: "parameters".
                parameters = self.update_parameters(parameters,gradients)

                #5 Compare the last and current cost
                current_cost = self.compute_cost(self.forward_propagation(X,parameters)["A2"],Y)
                if abs(last_cost-current_cost) < epsilon: break

                # Print the cost every 1000 iterations
                if cur_iterations % 10000 == 0:
                    print ("Cost after iteration %i: %f" %(cur_iterations, current_cost))
                cur_iterations +=1
                
            return parameters

        initial_parameters = self.initialize_parameters()
        parameters = gradient_descent(X,Y,n_iterations,initial_parameters)
        self.parameters = parameters








