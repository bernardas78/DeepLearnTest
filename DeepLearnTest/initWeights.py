import numpy as np

def initWeights(layerdims, alg="forRelu", debug=False):
    #initializes weights 
    #   e.g. layerdims = [5,4,3], where 5 = #input nodes; 3=#output nodes
    #   alg = "forRelu" = sqrt (2/n[l-1]) 
    #       = "xavier" = sqrt (1/n[l-1])
    #   Returns:
    #    dictionary (W1, b1, W2, b2, ..., WL, bL) , where L = len(layerdims)-1
    #       e.g. W1=[5,4] b1=[1,4],  W2=[3,4] b2=[1,3]

    params = {}
    for layer in range (1, len(layerdims)):
        n_l = layerdims[layer]
        n_l_minus = layerdims[layer-1]
        temp = np.random.rand(n_l_minus, n_l)
        if alg=="forRelu":
            params["W"+str(layer)] = temp * np.sqrt(2./n_l_minus)
        elif alg=="xavier":
            params["W"+str(layer)] = temp * np.sqrt(1./n_l_minus)
        else:
            raise Exception("initWeights.py failed: unknown alg")
        params["b"+str(layer)] = np.zeros((1,n_l), dtype="float64")
        if debug:
            print ("weights prior to scaling:",temp)
            print ("weights scaled:",params["W"+str(layer)])
    return params