import numpy as np

def initWeights(layerdims, activations, debug=False):
    #initializes weights 
    #   e.g. layerdims = [5,4,3], where 5 = #input nodes; 3=#output nodes
    #   activations: array size layerdims-1 (excl.input), elements one of the values:
    #       "relu" - use forRelu algorithm: sqrt (2/n[l-1]) 
    #       "tanh" - use "xavier" algorithm: sqrt (1/n[l-1])
    #       for others (e.g. softmax) - what to use?
    #   Returns:
    #    dictionary (W1, b1, W2, b2, ..., WL, bL) , where L = len(layerdims)-1
    #       e.g. W1=[4,5] b1=[4,1],  W2=[4,3] b2=[3,1]

    if debug:
        print ("==Starting initWeights.py")

    #seed() always creates new values; seed(1) reproduces same
    np.random.seed(1)
    #np.random.seed()

    params = {}
    for layer in range (1, len(layerdims)):
        n_l = layerdims[layer]
        n_l_minus = layerdims[layer-1]
        temp = np.random.randn(n_l, n_l_minus)
        if activations[layer-1]=="relu":
            params["W"+str(layer)] = temp * np.sqrt(2./n_l_minus)
        elif activations[layer-1]=="tanh":
            params["W"+str(layer)] = temp * np.sqrt(1./n_l_minus)
        else:
            params["W"+str(layer)] = temp * np.sqrt(2./n_l_minus)
        #    raise Exception("initWeights.py failed: unknown alg")
        params["b"+str(layer)] = np.zeros((n_l,1), dtype="float64")
        if debug:
            #print ("weights prior to scaling:",temp)
            #print ("weights scaled:",params["W"+str(layer)])
            print ("W"+str(layer)+".var=",np.var(params["W"+str(layer)]),"; .mean=",np.mean(params["W"+str(layer)]))

    if debug:
        print ("Returning keys:", params.keys())
        print ("==Ending initWeights.py")

    return params