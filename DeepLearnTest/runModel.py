import forwardProp as fp
import computeCost as cc
import backProp as bp
import updateParams as up

def runModel(iter_count, L, params, activations, X, y, learning_rate, debug=False):
    # Runs a model
    #   iter_count - iteration count
    #   L - number of layers (excl. input)
    #   params: dictionary of parameters; keys:
    #       W1,...,WL
    #       b1,...,bL
    #   activations: list of activation functions
    #   X: input of shape [nx,m]
    #   y: correct labels of shape [ny,m]; values 0 or 1
    #   learning_rate:
    #   debug:
    for iter in range (iter_count):
        za,yhat = fp.forwardProp(L, params, activations, X, debug=debug)
	    #yhat[:,0:5]
	    #za.keys()
	    #za["Z1"].shape
	    #za["A2"].shape

	    #compute cost
        cost = cc.computeCost(y, yhat, debug=debug)
        print("cost:",cost)

	    #run backprop
        grads = bp.backProp(L, activations, params, za, y, debug=debug)

	    #update params
        params = up.updateParams(L, params, grads, learning_rate)