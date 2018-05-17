import numpy as np
import forwardProp as fp
import computeCost as cc

def runModel(L, params, activations, X, y, debug=False, printcost=False):
    # Runs a model
    #   L - number of layers (excl. input)
    #   params: dictionary of parameters; keys:
    #       W1,...,WL
    #       b1,...,bL
    #   activations: list of activation functions, array of size L
    #   X: input of shape [nx,m]
    #   y: correct labels of shape [n_y,m]; values 0. or 1.
    #   debug:
    #   printcost: print costs and accuracy

    m = y.shape[1]

    _, yhat = fp.forwardProp(L, params, activations, X, regularization_technique="None", keep_prob=None, debug=debug)
    cost = cc.computeCost(y, yhat, debug=debug)
    accuracy = np.sum ( y [ np.argmax(yhat, axis=0), range(m) ] ) / m

    if printcost:
        print ("Final cost:", cost)
        print ("Final accuracy:", accuracy)

    return yhat, cost, accuracy