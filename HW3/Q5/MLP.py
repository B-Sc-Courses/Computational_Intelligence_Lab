import numpy as np
class MLP :
    '''
    # Multi Layer Perceptron 

    ## Introduction
    Provide:
    It prepares a network of perceptrons that is updated with a backpropagation algorithm.

    Nencessary for `__init__`:
    ```python
    feature_size:int, 
    output_size:int
    ```
    Optional:
    ```python
    *hidden_size # Number of perceptron in any layer
    ```
    ## Example:
    - `MLP(feature_size= 2, output_size= 3)` > Make One perceptron
    - `MLP(feature_size= 2, output_size= 3, 5, 6, 7)` > Make MLP
    '''
    
    def __init__(
            self,
            name: str,
            feature_size:int, 
            output_size:int,
            *hidden_size
        ) -> None:
        self.name = name
        # NN_size : A list of Net dimension sizes
        self.NN_size = [feature_size] + list(hidden_size) + [output_size]

        # layer_size: Number of layer 
        # (1: One perceptron, 2: One hidden layer, 3: Two hidden layer, ...)
        self.layer_size = len(self.NN_size)-1

        # Weights and Bias
        # actully, each index number is related to the layer number
        # Example: for 1 layer and without any hidden layer
        #          > W = [0, np.ndarray([feature_size, output_size])]
        #           >> W[1] : weight of first layer
        #          > B = [0, np.ndarray([output_size, 1])]
        #           >> B[1] : Bias of first layer
        self.W = [None,]     # List of weights
        self.B = [None,]     # List of Bias
        # Initializing the W & B
        self.initial_parameter()    
        # End of __init__
    
    def initial_parameter(self):
        '''
        ## initializing Weights and Bias
        - Call in `__init__`
        '''
        for i in range(self.layer_size):
            self.W.append(np.random.random([self.NN_size[i], self.NN_size[i+1]]))
            self.B.append(np.zeros((self.NN_size[i+1], 1)))
        # End of initial_parameter

    def sigmoid(self, z: np.ndarray) :
        '''
        ## Calculating sigmoid(Bipolar) 
         (-1< output <1)
        - output = 2/(1+e^-x) -1
        '''
        return (2 / (1 + np.exp(-z))) - 1
        # End of Sigmoid
    
    def sigmoid_der(self, a: np.ndarray=None, z: np.ndarray=None):
        '''
        ## Derivative of Bipolar Sigmoid
        - if the input `a` is not empty and it is `numpy array`
            - return `(1-a^2)/2`
        - if the input `a` is empty but `z` is not empty and it is `numpy array`
            - calculate the value of `a`
            - return `(1-a^2)/2`
        - if both of a and b are empty
            - Error!
        '''
        if isinstance(a, np.ndarray) is not True: 
            if isinstance(z, np.ndarray):
                a = self.sigmoid(z)
            else:
                raise Exception("Both of a and z are None!")
        return (1 - a**2) / 2
        # End of Derivative of Sigmoid(Bipolar)
        
    def forward(self, x: np.ndarray):
        '''
        ## Forward (Predict) function
        - `a0` = input data (`x`)
        - z_L = W_L.T @ a_(L-1) + B_L
        - a_L = sigmoid(z_L)
        - return the output of NN
        '''
        a = [np.copy(x)] # a_0 = input data
        z = [None]          # z_0 : NO z_0 so an None is stored as z_0
        for i in range(self.layer_size):
            z.append(self.W[i+1].T @ a[i] + self.B[i+1])
            a.append(self.sigmoid(z[i+1]))
        
        return a[-1]
        # End of Forward or Predict fucntion
        
    def gradient(self, x: np.ndarray, y: np.ndarray):
        '''
        ## Gradient
        Becuz of the use Backpropagation Algo, dW and dB must calculated by Gradient!  
        '''
        # Initializing a, z, error, delta, gradW gradB matrix
        #region
        a = [np.copy(x)] # a_0 = input data
        z = [None]          # z_0 : NO z_0 so an None is stored as z_0
        error_layer = [None] * (self.layer_size+1)
        delta_layer = [None] * (self.layer_size+1)
        grad_W = [None] * (self.layer_size+1)
        grad_B = [None] * (self.layer_size+1)
        #endregion
        
        # Calculating z_L and a_L {a_0 = x and z_0, B_0, W_0 are None}
        for i in range(self.layer_size):
            z.append((self.W[i+1].T @ a[i]) + self.B[i+1])
            a.append(self.sigmoid(z[i+1]))
        
        # Caclulating the backpropagation for the outhermost layer (the last)!
        #region
        # Error
        error_layer[self.layer_size] = y - a[self.layer_size]
        # delta
        delta_layer[self.layer_size] = self.sigmoid_der(a[self.layer_size]) \
                                            * error_layer[self.layer_size]
        # gradient W
        grad_W[self.layer_size] = a[self.layer_size-1] \
                                            @ delta_layer[self.layer_size].T
        # gradient B
        grad_B[self.layer_size] = np.sum(delta_layer[self.layer_size], axis=1).reshape(-1, 1)
        # End of Calculation for the outhermost layer (last layer)
        #endregion
        
        # Calculation fot hidden layer if there is :)
        for i in range(self.layer_size-1, 0, -1):
            error_layer[i] = self.W[i+1] @ delta_layer[i+1]
            delta_layer[i] = self.sigmoid_der(a[i]) * error_layer[i]
            grad_W[i] = a[i-1] @ delta_layer[i].T
            grad_B[i] = np.sum(delta_layer[i], axis=1).reshape((self.NN_size[i], 1))
        
        return grad_W, grad_B
        # End of Gradient Function

    def fit(
            self, 
            x: np.ndarray, 
            y: np.ndarray, 
            learning_rate: float=0.01, 
            epoches: int=10
        ) :
        '''
        ## Training Function
        - input:
            - x: data input
            - y: label
            - learning_rate: etta -> default is 0.01
            - epoches: -> default is 10
        - get grad W and grad B from `grdient function`
        - update Weights and Bias
        '''
        for i in range(epoches):
            grad_W, grad_B = self.gradient(x, y)
            for j in range(self.layer_size):
                self.W[j+1] += learning_rate * grad_W[j+1]
                self.B[j+1] = self.B[j+1] + learning_rate * grad_B[j+1]
        
        # return self.forward(x)
    # End of Train!


