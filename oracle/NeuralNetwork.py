#encoding=utf-8
import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - np.tanh(x)*np.tanh(x)

def logistic(x):
    return 1/(1+np.exp(-x))

def logistic_derivative(x):
    return logistic(x)*(1-logistic(x))

class  NeuralNetwork:
    # self 相当于Java中的this
    def __init__(self,layers, activation='tanh'):
        if activation ==  'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative()
