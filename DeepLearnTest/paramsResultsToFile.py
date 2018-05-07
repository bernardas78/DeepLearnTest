import numpy as np

def paramsResultsToFile(iter_count, L, params, activations, learning_rate,\
    optimization_technique,beta_momentum, beta_rmsprop, cost, accuracy):

    layer_sizes = np.zeros((L),'int')
    for layer in range(0, L):
         layer_sizes[layer] = params["b"+str(layer+1)].shape[0]

    file = open("..\\results.csv","a")

    strToWrite = str(iter_count) + ";" + \
        str(L) + ";" + \
        str(layer_sizes) + ";" + \
        str(activations) + ";" + \
        str(learning_rate) + ";" + \
        str(optimization_technique) + ";" + \
        str(beta_momentum) + ";" + \
        str(beta_rmsprop) + ";" + \
        str(cost) + ";" + \
        str(accuracy) + "\n"
    file.write(strToWrite)

    file.close() 