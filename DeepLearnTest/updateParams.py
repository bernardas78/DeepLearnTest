import numpy as np

def updateParams(L, params, grads, learning_rate):
    #   L - layer count (excl.input layer)
    #   params: dictionary of weights, contains keys:
    #     W1,...,WL - where L number of layers
    #     b1,...,bL
    #   grads: gradients of params dictionary, contains keys:
    #     dW1,...,dWL 
    #     db1,...,dbL
    #   learing_rate 
    #
    # Returns: updated dictionary of params:
    #     W1,...,WL - where L number of layers
    #     b1,...,bL
    for layer in range (0, L):
        W = params["W"+str(layer+1)]
        b = params["b"+str(layer+1)]
        dW = grads["dW"+str(layer+1)]
        db = grads["db"+str(layer+1)]

        W -= dW * learning_rate
        b -= db * learning_rate

        params["W"+str(layer+1)] = W
        params["b"+str(layer+1)] = b

    return params