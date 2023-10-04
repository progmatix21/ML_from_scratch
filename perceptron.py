#!/usr/bin/env python3

"""
Module to implement a single perceptron and a
helper class to generate training data.

Reference implementation: Create a D2A converter using a 
single perceptron.
"""
import numpy as np
import matplotlib.pyplot as plt

class Perceptron():
    """
    Class to implement an n-input perceptron with identity activation.
    
    Arguments:
        n_inputs: no. of inputs
        
    Attributes:
        _weights: list of weights(w0,...,wn), one per epoch
        _cost: list of cost function values(one per epoch)
        _epochs: no. of epochs to train the perceptron
        
    Methods:
        fit(X,y,n_epochs=10,batch_size=1): train the perceptron with
            training data X,y for n_epochs(default 10) with batch size
            of batch_size(default 1)
        predict(X): predict y for test data X on an already fitted perceptron.
            
    Since the training  and test data are deterministic,
    we do not need cross-validation.  It is enough if we
    just look at the cost function.
"""

    def __init__(self, n_inputs):
        """Create perceptron with n inputs."""
        
        self._n_inputs = n_inputs
        # Generate n_inputs random weights[0,1) and 1 bias.
        self._weights = np.random.uniform(low=-0.1, high=0.1, size=(n_inputs+1,))
        # Will get updated at end of every epoch.  At end of training, has all
        # weights of a trained perceptron.
        # history of weights
        self._weights_hist = []
        # store the initial weights first
        self._weights_hist.append(self._weights)
        
        # Array of costs 1 per epoch.
        self._cost_hist = []
        
        # For how many epochs have we trained?
        self._epochs = 0
                
        
    def fit(self,X,y, epochs=10, batch_size=1):
        """Train the perceptron with X,y
        
        Arguments: 
            epochs: no.of epochs to train
            batch_size: batch size            
        """
        lr = 0.01  #learning rate
        
        for i in range(epochs):
            for x_vec,y_true in zip(X,y):
                x_vec = np.append([1],x_vec) # Prepend 1 for the bias
                output = np.dot(self._weights,x_vec).squeeze()
                
                #Calculate delta_w from error gradient and learning rate
                ######### Update the weights ############
                delta_w = lr*x_vec*(y_true-output)
                self._weights = self._weights+delta_w

            #Keep track of the cost function
            cost_func = 0.5*(output-y_true)**2
            self._cost_hist.append(cost_func)
                            
            #keep weights history
            self._weights_hist.append(self._weights)
                
            self._epochs += 1  #Increment epochs count
            
        #Return training history
        return {"cost":self._cost_hist, "weights":self._weights_hist,"epochs":self._epochs}
        
    def predict(self,X):
        """
        Predict target y for test data X
        Arguments: X test data
        """
        #Insert a 1 in the first column(index 0) of all X rows
        _X = np.insert(X, 0, 1, axis=1)
        return np.dot(_X,self._weights)
        
    def __repr__(self):
        """
        Print representation of the object.
        """
        return f'perceptron <{id(self):x}> inputs:{self._n_inputs}, epochs:{self._epochs}, weights:{self._weights}'


class Vector_generator():
    """ 
    Generate list of binary digit vectors of specified length and width.
    
    Arguments:
        n_bits: width of the vector(default 8).
        
    Methods:
        get_samples(n_samples=10): get specified number of random
            samples(default 10) of type X,y with y scaled[0,1)
        get_cons_samples(): get consecutive samples from 0 to 2^n_bits-1        
    """

    def __init__(self, n_bits=8):
        """Initialize a vector generator of width n_bits"""
        self._n_bits = n_bits
        self._X_arr = []
        self._y_arr = []
        
    def get_samples(self, n_samples=10):
        """Get specified no. of random vectors(default 10)"""
        
        self._X_arr = []
        self._y_arr = []
        
        for i in range(n_samples):
            # Get 1 random integer in range 0,2**n_bits
            n = np.random.randint(2**self._n_bits)
            # Convert it to binary vector
            X = [int(i) for i in list(f'{n:0{self._n_bits}b}')]
            # Alternatively
            # X = np.array(list(f'{n:0{_n_bits}b}')).astype(int)
            # Its equivalent floating point value in range [0,1)
            y = round(n/(2**self._n_bits),6)
            self._X_arr.append(X)
            self._y_arr.append(y)
            
        return self._X_arr, self._y_arr

        
    def get_cons_samples(self):
        """get consecutive samples from 0 to 2^n_bits-1"""
        
        self._X_arr = []
        self._y_arr = []
        
        for cons_n in range(2**self._n_bits):
            # Get the integer from the range 0,2**n_bits
            # Convert it to binary vector
            X = [int(i) for i in list(f'{cons_n:0{self._n_bits}b}')]
            y = round(cons_n/(2**self._n_bits),6)
            self._X_arr.append(X)
            self._y_arr.append(y)
            
        return self._X_arr, self._y_arr



if __name__ == "__main__":
    # Create a data generator
    v_gen10bit = Vector_generator(n_bits = 10)    
    X,y = v_gen10bit.get_samples(n_samples=200)
    
    #Create the perceptron
    percep = Perceptron(n_inputs = 10)
    print(percep)
    
    #Train the perceptron
    training_history = percep.fit(X,y,epochs=50)
    print(percep)

    #plot the minimization of the cost function and ...    
    #...the convergence of the weights+bias
    #In short, plot the training progress.
    fig,ax = plt.subplots(1,3, figsize=(15,5))
    
    # Some nuances of plotting. For plotting costs, epochs start at 1
    # Cost at the end of each epoch    
    ax[0].plot(np.arange(1,training_history["epochs"]+1),training_history["cost"], '.-')
    ax[0].set_xticks(np.arange(0,training_history["epochs"]+1,5))
    ax[0].set_title("Cost function vs. Epoch")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Cost")
    ax[0].grid()
    

    # There is one more set of weights than number of epochs because
    # We also plot the initial weights before we started the first epoch
    # Epoch starts with 0

    ax[1].plot(training_history["weights"], '.-')
    ax[1].set_xticks(np.arange(0,training_history["epochs"]+1,5))
    ax[1].set_title("Weight vs. Epoch")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Weights")
    ax[1].grid()

    #Try out predictions with our perceptron
    #Generate consecutive data
    Xt,yt = v_gen10bit.get_cons_samples()
    y_pred = percep.predict(Xt)
    
    #Plot y_pred vs. y_true
    ax[2].scatter(yt, y_pred, c = 'c', alpha=0.5, label='predicted')
    ax[2].plot(yt,yt, 'k', label='identity')
    ax[2].set_title("y_pred vs. y_true scaled [0,1]")
    ax[2].set_xlabel("y_true")
    ax[2].set_ylabel("y_pred")
    ax[2].legend(loc='best')
    ax[2].grid()
    plt.tight_layout()
    
    
    plt.show()

    
    
