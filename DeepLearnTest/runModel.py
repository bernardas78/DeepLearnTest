import numpy as np
import matplotlib.pyplot as plt 
import forwardProp as fp
import computeCost as cc
import backProp as bp
import updateParams as up
import time
import paramsResultsToFile as prtf

def runModel(iter_count, L, params, activations, X, y, learning_rate, minibach_size=None,\
    optimization_technique="GradientDescent",beta_momentum=None, beta_rmsprop=None,\
    debug=False, printcost=False):
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
    #   minibach_size: size of a minibatch to train on; if not set - then set to full data set's size
    #   optimization_technique: way of how params (W's and b's) are updated
    #       one of ["GradientDescent","GradientDescentWithMomentum","RMSProp","Adam"]
    #   beta_momentum: opt. techniques GradientDescentWithMomentum, Adam; optimization parameter beta1
    #   beta_rmsprop: opt. techniques RMSProp, Adam; optimization parameter beta2 (exp.weighted square averages)
    #   debug:
    #   printcost: plot costs or not

    m = y.shape[1]
    if minibach_size is None:
        minibach_size = m

    accuracies = []
    costs = []

    #store optimization parameter values in previously calculated iterations
    opt_params = {}

    #fp_time = 0.
    #cc_time = 0.
    #bp_time = 0.
    #up_time = 0.

    if printcost:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        mline, = ax.plot([], costs, 'b-')

    for iter in range (iter_count):
        if debug:
            print ("==========ITER:", str(iter))

        #set current minibatch
        iterations_per_epoch = int (np.ceil ( m / minibach_size))
        iteration_within_epoch = np.mod ( iter, iterations_per_epoch)
        smpl_start = iteration_within_epoch * minibach_size
        smpl_end = np.minimum (m, (iteration_within_epoch + 1) * minibach_size )
        X_minib = X[:,smpl_start:smpl_end]
        y_minib = y[:,smpl_start:smpl_end]
        m_minib = smpl_end - smpl_start
        #print ("minib:[",smpl_start,",",smpl_end,"]",sep="")

        #Forward prop
        #start = time.perf_counter()
        #za,yhat = fp.forwardProp(L, params, activations, X, debug=debug)
        za,yhat = fp.forwardProp(L, params, activations, X_minib, debug=debug)
        #fp_time += time.perf_counter() - start

	    #Compute cost,accuracy
        #start = time.perf_counter()
        #cost = cc.computeCost(y, yhat, debug=debug)
        cost = cc.computeCost(y_minib, yhat, debug=debug)
        #cc_time += time.perf_counter() - start
        #accuracy = np.sum ( y [ np.argmax(yhat, axis=0), range(m) ] ) / m
        accuracy = np.sum ( y_minib [ np.argmax(yhat, axis=0), range(m_minib) ] ) / m_minib

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


	    #Backprop
        #start = time.perf_counter()
        grads = bp.backProp(L, activations, params, za, y_minib, debug=debug)
        #bp_time += time.perf_counter() - start

	    #Update params
        #start = time.perf_counter()
        #print ("opt_params.keys() BEFORE:",opt_params.keys())
        params, opt_params = up.updateParams(L, params, grads, learning_rate,\
            optimization_technique, beta_momentum, beta_rmsprop, opt_params, debug=debug)
        #print ("opt_params.keys() AFTER:",opt_params.keys())
        #up_time += time.perf_counter() - start

    #print ("fp_time:",fp_time)
    #print ("cc_time:",cc_time)
    #print ("bp_time:",bp_time)
    #print ("up_time:",up_time)
    if debug:
        print ("Final cost:", cost)
        print ("Final accuracy:", accuracy)

    #Compute cost, accuracy for entire training data set
    if True:
        _, yhatFull = fp.forwardProp(L, params, activations, X, debug=False)
        costFull = cc.computeCost(y, yhatFull, debug=debug)
        accuracyFull = np.sum ( y [ np.argmax(yhatFull, axis=0), range(m) ] ) / m
    if debug:
        print ("Final cost on full training set:", costFull)
        print ("Final accuracy on full training set:", accuracyFull)

    #output parameters and results to a file
    if True:
        prtf.paramsResultsToFile(iter_count, L, params, activations, learning_rate,\
            optimization_technique,beta_momentum, beta_rmsprop, costFull, accuracyFull)

    #if printcost:
    #    plt.plot(range(len(costs)), costs)
    #    plt.show()

    return yhatFull, grads, params