#!/usr/bin/env python3

"""
Module to implement a single perceptron and a
helper class to generate training data.

Reference implementation: Create a D2A converter using a 
single perceptron.
"""
import numpy as np

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
    
    """

    def __init__(self, n_inputs):
        """Create perceptron with n inputs."""
        _weights = None
        _cost = None
        _epochs = None
        
        pass
        
    def fit(X,y, epochs=10, batch_size=1):
        """Train the perceptron with X,y
        
        Arguments: 
            epochs: no.of epochs to train
            batch_size: batch size            
        """
        
        pass
        
    def predict(X):
        """
        Predict target y for test data X
        Arguments: X test data
        """


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


"""
1. Generate training data
2. Create perceptron class
3. Create neural network and train
4. Predict using test data
5. Evaluate performance(cost function curve)

Since the training  and test data are deterministic,
we do not need cross-validation.  It is enough if we
just look at the cost function.
"""


if __name__ == "__main__":
    #training_data = generate_training_data(n_bits=8,n_samples=10)
    
    # Create a data generator
    v_gen8bit = Vector_generator(n_bits = 8)
    print("training data= ", v_gen8bit.get_samples(n_samples=5))
    print("-"*20)
    print("training data= ", v_gen8bit.get_samples(n_samples=10))    
    #print("training data= ", v_gen8bit.get_cons_samples())    

