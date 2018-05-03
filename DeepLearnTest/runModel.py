import numpy as np
import matplotlib.pyplot as plt 
import forwardProp as fp
import computeCost as cc
import backProp as bp
import updateParams as up
import time
import copy

#Starting changes for comparing several optimization techniques
def runModel(iter_count, L, params, activations, X, y, learning_rate, minibach_size=None,\
    optimization_technique1="GradientDescent",beta_momentum1=None,\
    optimization_technique2="GradientDescent",beta_momentum2=None,\
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
    #   beta_momentum: opt. technique GradientDescentWithMomentum, optimization parameter beta
    #   debug:
    #   printcost: plot costs or not

    m = y.shape[1]
    if minibach_size is None:
        minibach_size = m

    accuracies1 = []
    accuracies2 = []
    costs1 = []
    costs2 = []

    #store optimization parameter values in previously calculated iterations
    opt_params1 = {}
    opt_params2 = {}

    #set both techniques' initial param values, optimize separately
    params1 = copy.deepcopy(params)
    params2 = copy.deepcopy(params)

    #fp_time = 0.
    #cc_time = 0.
    #bp_time = 0.
    #up_time = 0.

    if printcost:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        mline1, = ax.plot([], costs1, 'b-')
        mline2, = ax.plot([], costs2, 'r-')

    for iter in range (iter_count):
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
        #for the sake of comparing 2 optimization techniques, use the same minibatch (so that graph does not jump)
        #X_minib = X[:,0:minibach_size]
        #y_minib = y[:,0:minibach_size]
        #m_minib = minibach_size


        #Forward prop
        #start = time.perf_counter()
        #za,yhat = fp.forwardProp(L, params, activations, X, debug=debug)
        za1,yhat1 = fp.forwardProp(L, params1, activations, X_minib, debug=debug)
        za2,yhat2 = fp.forwardProp(L, params2, activations, X_minib, debug=debug)
        #fp_time += time.perf_counter() - start

	    #Compute cost,accuracy
        #start = time.perf_counter()
        #cost = cc.computeCost(y, yhat, debug=debug)
        cost1 = cc.computeCost(y_minib, yhat1, debug=debug)
        cost2 = cc.computeCost(y_minib, yhat2, debug=debug)
        #cc_time += time.perf_counter() - start
        #accuracy = np.sum ( y [ np.argmax(yhat, axis=0), range(m) ] ) / m
        accuracy1 = np.sum ( y_minib [ np.argmax(yhat1, axis=0), range(m_minib) ] ) / m_minib
        accuracy2 = np.sum ( y_minib [ np.argmax(yhat2, axis=0), range(m_minib) ] ) / m_minib

        accuracies1 = np.append(accuracies1, accuracy1)
        accuracies2 = np.append(accuracies2, accuracy2)
        costs1 = np.append(costs1, cost1)
        costs2 = np.append(costs2, cost2)

        if printcost:
            #Draw costs
            mline1.set_xdata( np.linspace(1,len(costs1),len(costs1)) )
            mline1.set_ydata(costs1)
            mline2.set_xdata( np.linspace(1,len(costs2),len(costs2)) )
            mline2.set_ydata(costs2)
            ax.autoscale_view()
            ax.relim()
            fig.canvas.draw()
            fig.canvas.flush_events()


	    #Backprop
        #start = time.perf_counter()
        grads1 = bp.backProp(L, activations, params1, za1, y_minib, debug=debug)
        grads2 = bp.backProp(L, activations, params2, za2, y_minib, debug=debug)
        #bp_time += time.perf_counter() - start

	    #Update params
        #start = time.perf_counter()
        #print ("opt_params.keys() BEFORE:",opt_params.keys())
        params1, opt_params1 = up.updateParams(L, params1, grads1, learning_rate,\
            optimization_technique1, beta_momentum1, opt_params1, debug=debug)
        params2, opt_params2 = up.updateParams(L, params2, grads2, learning_rate,\
            optimization_technique2, beta_momentum2, opt_params2, debug=debug)
        #print ("opt_params.keys() AFTER:",opt_params.keys())
        #up_time += time.perf_counter() - start

    #print ("fp_time:",fp_time)
    #print ("cc_time:",cc_time)
    #print ("bp_time:",bp_time)
    #print ("up_time:",up_time)
    print ("Final cost1:", cost1)
    print ("Final cost2:", cost2)
    print ("Final accuracy1:", accuracy1)
    print ("Final accuracy2:", accuracy2)
    #if printcost:
    #    plt.plot(range(len(costs)), costs)
    #    plt.show()

    return za1, yhat1, grads1, params1