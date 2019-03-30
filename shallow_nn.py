import numpy as np

#两层神经网络 - 包含一个隐藏层
class ShallowNN:
    def __init__(self,n_x,n_h,n_y):
        self.n_x = n_x
        self.n_h = n_h
        self.n_y = n_y
        # n_x -- the size of the input layer  n_x = X.shape[0]
        # n_h -- the size of the hidden layer  n_h = 4
        # n_y -- the size of the output layer  n_y = Y.shape[0] 
        self.parameters = dict()  
    
    def initialize_parameters(self):
        W1 = np.random.randn(self.n_h,self.n_x) * 0.01
        b1 = np.zeros((self.n_h,1))
        W2 = np.random.randn(self.n_y,self.n_h) * 0.01
        b2 = np.zeros((self.n_y,1))

        assert (W1.shape == (self.n_h, self.n_x))
        assert (b1.shape == (self.n_h, 1))
        assert (W2.shape == (self.n_y, self.n_h))
        assert (b2.shape == (self.n_y, 1))
        
        initial_parameters = {"W1": W1,"b1": b1,"W2": W2,"b2": b2}
        return initial_parameters

    def sigmoid(self,y_hat):
        return 1/(1+np.exp(y_hat))
    
    def forward_propagation(self,X,parameters):
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']

        Z1 = np.dot(W1, X) + b1
        #Using tanh activate function for hidden layer
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        #Using sigmoid activate function for output layer
        A2 = self.sigmoid(Z2)

        assert(A2.shape == (1, X.shape[1]))

        cache = {"Z1": Z1,"A1": A1,"Z2": Z2,"A2": A2}
        return cache

    def compute_cost(self,y_hat,y):
        m = y.shape[1]
        return -np.sum(y*np.log(y_hat)+(1-y)*np.log(1-y_hat)) / m
    
    def backward_propagation(self,parameters,cache,X,y):
        """
        Implement the backward propagation
        
        Arguments:
        parameters -- python dictionary containing our parameters 
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
        X -- input data of shape (2, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)
        
        Returns:
        gradients -- python dictionary containing your gradients with respect to different parameters
        """
        m = X.shape[1]
        W1 = parameters['W1']
        W2 = parameters['W2']

        A1 = cache['A1']
        A2 = cache['A2']

        dZ2 = A2 - y
        dW2 = 1/m * np.dot(dZ2, A1.T)
        db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
        dW1 = 1/m * np.dot(dZ1, X.T)
        db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)

        gradients = {"dW1": dW1,"db1": db1,"dW2": dW2,"db2": db2}
        return gradients
    
    def update_parameters(self,parameters, gradients, learning_rate = 1.2):
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']
        
        dW1 = gradients["dW1"]
        db1 = gradients["db1"]
        dW2 = gradients["dW2"]
        db2 = gradients["db2"]

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        
        new_parameters = {"W1": W1,"b1": b1,"W2": W2,"b2": b2}
        return new_parameters

    def fit(self,X,y,n_iterations = 1e5):
        """  
        Arguments:       
        X -- dataset of shape (num of features, number of examples)
        Y -- labels of shape (1, number of examples)
        num_iterations -- Number of iterations in gradient descent loop 
        """
        def gradient_descent(X,y,n_iterations,initial_parameters,epsilon = 1e-7):
            cur_iterations = 0
            parametrs = dict()
            parameters = {"W1": initial_parameters['W1'],"b1": initial_parameters['b1'],"W2": initial_parameters['W2'],"b2": initial_parameters['b2']}

            while cur_iterations < n_iterations:
                #1 Forward_propagation to get the essential data for backward_propagation 
                #cache:dict  {"Z1": Z1,"A1": A1,"Z2": Z2,"A2": A2} 
                cache = self.forward_propagation(X,parameters)

                #2 Get the cost for current W and b (int:cost)  
                #last_cost means cost before the descent
                last_cost = self.compute_cost(cache["A2"],y)

                #3 Backward_propagation to calculate the gradient for W and b
                #gradients:dict   {"dW1":dW1,"dW2":dW2,"db1":db1,"db2":db2}
                gradients = self.backward_propagation(parameters,cache,X,y)

                #4 Gradient descent parameter update  Inputs: "parameters, grads". Outputs: "parameters".
                parameters = self.update_parameters(parameters, gradients)

                #5 Compare the last and current cost
                current_cost = self.compute_cost(self.forward_propagation(X,parameters)["A2"],y)
                if abs(last_cost-current_cost) < epsilon: break

                # Print the cost every 1000 iterations
                if cur_iterations % 10000 == 0:
                    print ("Cost after iteration %i: %f" %(cur_iterations, current_cost))

                cur_iterations +=1
            
            return parameters

        initial_parameters = self.initialize_parameters()
        parameters = gradient_descent(X,y,n_iterations,initial_parameters)
        self.parameters = parameters
    
    def predict(self,X):
        cache = self.forward_propagation(X,self.parameters)
        y_hat = cache["A2"]
        predictions = np.array( [1 if x >0.5 else 0 for x in y_hat.reshape(-1,1)] ).reshape(y_hat.shape)

        return predictions

        

        

