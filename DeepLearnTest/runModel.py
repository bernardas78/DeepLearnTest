import numpy as np
import matplotlib.pyplot as plt 
import forwardProp as fp
import computeCost as cc
import backProp as bp
import updateParams as up

def runModel(iter_count, L, params, activations, X, y, learning_rate, debug=False, printcost=False):
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

    m = y.shape[1]

    accuracies = []
    costs = []
    plt.ion()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    mline, = ax.plot([], costs, 'b-')

    for iter in range (iter_count):
        za,yhat = fp.forwardProp(L, params, activations, X, debug=debug)
	    #yhat[:,0:5]
	    #za.keys()
	    #za["Z1"].shape
	    #za["A2"].shape

	    #compute cost
        cost = cc.computeCost(y, yhat, debug=debug)
        if printcost:
            #compute accuracy
            accuracy = np.sum ( y [ np.argmax(yhat, axis=0), range(m) ] ) / m
            print ("iter:", str(iter), "; cost:", cost, "; accuracy:", accuracy)

            accuracies = np.append(accuracies, accuracy)
            costs = np.append(costs, cost)
            #print(costs)

            mline.set_xdata( np.linspace(1,len(costs),len(costs)) )
            mline.set_ydata(costs)

            ax.autoscale_view()
            ax.relim()

            fig.canvas.draw()
            fig.canvas.flush_events()


	    #run backprop
        grads = bp.backProp(L, activations, params, za, y, debug=debug)

	    #update params
        params = up.updateParams(L, params, grads, learning_rate)

    #if printcost:
    #    plt.plot(range(len(costs)), costs)
    #    plt.show()