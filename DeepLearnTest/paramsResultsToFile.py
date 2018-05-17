import numpy as np
import datetime

def paramsResultsToFile(iter_count, L, params, activations, learning_rate,\
    optimization_technique,beta_momentum, beta_rmsprop, regularization_technique, lambd, keep_prob,\
    costTrain, accuracyTrain, costTest, accuracyTest):
    #Output model parameters and results to a results.csv
    #   iter_count - number of iterations model ran
    #   L - number of layers excl. input
    #   params - W's and b's of the model (not written to file)
    #       layer_sizes - written to file from params' info
    #   activations - activation function
    #   learning_rate
    #   optimization_technique: way of how params (W's and b's) are updated
    #       one of ["GradientDescent","GradientDescentWithMomentum","RMSProp","Adam"]
    #   beta_momentum: opt. techniques GradientDescentWithMomentum, Adam; optimization parameter beta1
    #   beta_rmsprop: opt. techniques RMSProp, Adam; optimization parameter beta2 (exp.weighted square averages)
    #   regularization_technique
    #   lambd: lambda, hyperparameter for regularization technique L2
    #   keep_prob: vector size L, used for regularization technique Dropout
    #   costTrain
    #   accuracyTrain 
    #   costTest
    #   accuracyTest

    layer_sizes = np.zeros((L),'int')
    for layer in range(0, L):
         layer_sizes[layer] = params["b"+str(layer+1)].shape[0]

    file = open("..\\results.csv","a")

    strToWrite = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ";" + \
        str(iter_count) + ";" + \
        str(L) + ";" + \
        str(layer_sizes) + ";" + \
        str(activations) + ";" + \
        str(learning_rate) + ";" + \
        str(optimization_technique) + ";" + \
        str(beta_momentum) + ";" + \
        str(beta_rmsprop) + ";" + \
        regularization_technique + ";" + \
        str(lambd) + ";" + \
        str(keep_prob) + ";" + \
        str(costTrain) + ";" + \
        str(accuracyTrain) + ";" + \
        str(costTest) + ";" + \
        str(accuracyTest) + "\n"
    file.write(strToWrite)

    file.close() 