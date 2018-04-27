import numpy as np
import matplotlib.pyplot as plt 
import forwardProp as fp
import computeCost as cc
import backProp as bp
import updateParams as up
import time

def runModel(iter_count, L, params, activations, X, y, learning_rate, debug=False, printcost=False):
    # Runs a model
    #   iter_count - iteration count
    #   L - number of layers (excl. input)
    #   params: dictionary of parameters; keys:
    #       W1,...,WL
    #       b1,...,bL
    #   activations: list of activation functions, array of size L
    #   X: input of shape [nx,m]
    #   y: correct labels of shape [n_y,m]; values 0. or 1.
    #   learning_rate:
    #   debug:

    m = y.shape[1]

    accuracies = []
    costs = []

    fp_time = 0.
    cc_time = 0.
    bp_time = 0.
    up_time = 0.

    if printcost:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        mline, = ax.plot([], costs, 'b-')

    for iter in range (iter_count):
        print ("==========ITER:", str(iter))
        start = time.perf_counter()
        za,yhat = fp.forwardProp(L, params, activations, X, debug=debug)
        fp_time += time.perf_counter() - start
	    #yhat[:,0:5]
	    #za.keys()
	    #za["Z1"].shape
	    #za["A2"].shape

	    #compute cost,accuracy
        start = time.perf_counter()
        cost = cc.computeCost(y, yhat, debug=debug)
        cc_time += time.perf_counter() - start
        accuracy = np.sum ( y [ np.argmax(yhat, axis=0), range(m) ] ) / m

        accuracies = np.append(accuracies, accuracy)
        costs = np.append(costs, cost)

        if printcost:
            #Draw costs
            mline.set_xdata( np.linspace(1,len(costs),len(costs)) )
            mline.set_ydata(costs)
            ax.autoscale_view()
            ax.relim()
            fig.canvas.draw()
            fig.canvas.flush_events()


	    #run backprop
        start = time.perf_counter()
        grads = bp.backProp(L, activations, params, za, y, debug=debug)
        bp_time += time.perf_counter() - start

	    #update params
        start = time.perf_counter()
        params = up.updateParams(L, params, grads, learning_rate, debug=debug)
        up_time += time.perf_counter() - start

    print ("fp_time:",fp_time)
    print ("cc_time:",cc_time)
    print ("bp_time:",bp_time)
    print ("up_time:",up_time)
    print ("Final cost:", cost)
    print ("Final accuracy:", accuracy)
    #if printcost:
    #    plt.plot(range(len(costs)), costs)
    #    plt.show()

    return za, yhat, grads, params