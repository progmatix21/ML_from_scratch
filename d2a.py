#!/usr/bin/env python3

import numpy as np


'''
Create a D2A converter using a single perceptron.
1. Generate training data
2. Create perceptron class
3. Create neural network and train
4. Predict using test data
5. Evaluate performance(cost function curve)

Since the training  and test data are deterministic,
we do not need cross-validation.  It is enough if we
just look at the cost function.
'''

def generate_training_data(n_bits=8,n_samples=10):
    '''
    1. Generate a random integer.
    2. Convert the integer to binary.
    3. Get the binary digits as a vector(X)
    4. Divide the integer by 255(y).
    '''
    
    #Create a generator for random integers
    rand_int_gen = (np.random.randint(2**n_bits-1) for i in range(n_samples))
    
    #initialize 2 arrays to return
    X_arr = []
    y_arr = []
    
    for r in rand_int_gen:
        X_arr.append([int(i) for i in list(f'{r:0{n_bits}b}')])
        y_arr.append(round(r/255,4))

    '''    
    for i in range(n_samples):
        n = np.random.randint(2**n_bits-1)
        X = [int(i) for i in list(f'{n:0{n_bits}b}')]
        y = round(n/255,4)
        X_arr.append(X)
        y_arr.append(y)
    '''    
    return X_arr,y_arr
    

if __name__ == "__main__":
    training_data = generate_training_data(n_bits=8,n_samples=10)
    print("training data= ", training_data)

