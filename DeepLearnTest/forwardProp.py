import numpy as np

def forwardProp(L, params, activations, X, debug=False, printall=False):
    #caclulates forward propagation of a neural network
    #   L - layer count (excl.input layer)
    #   params - dictionary containing "W1", "b1",..."WL", "bL" parameters
    #   activations - array of size L of activation functions, each of one of the values:
    #       "sigmoid"
    #       "tanh"
    #       "relu"
    #       "softmax"
    #   X - import matrix of shape [nx,m]
    #Returns: aL (yhat) of size [ny,m]
    #         za - Z's and A'ctivations of intermediate steps (used in backprop). Incl. A0 (input)

    if debug:
        print ("==Starting forwardProp.py")

    za = {}
    za["A0"] = X
    al_prev = X
    for layer in range (0, L):
        Wl = params["W"+str(layer+1)]
        bl = params["b"+str(layer+1)]
        activation = activations[layer]
        if debug:
            print ("W",str(layer+1),".shape:",Wl.shape,\
                "; b",str(layer+1),".shape:",bl.shape,\
                "; A",str(layer),".shape:",al_prev.shape,\
                "; activation:",activation,sep="")
            #print (,sep="")
        zl = np.add ( np.dot ( Wl, al_prev ), bl )
        if activation == "sigmoid":
            al = 1. / ( 1. + np.exp (-zl))
        elif activation == "tanh":
            al = np.tanh ( zl )
        elif activation == "relu":
            al = np.fmax (zl, np.zeros(zl.shape, dtype="float64"))
        elif activation == "softmax":
            al = np.exp (zl) / np.sum(np.exp (zl), axis=0, keepdims=True)
        else:
            raise Exception("forwardProp.py failed: unknown activation")

        za["Z"+str(layer+1)] = zl
        za["A"+str(layer+1)] = al

        if debug:
            print ("Z",str(layer+1),".shape:",zl.shape,"; A",str(layer+1),".shape:",al.shape,sep="")
        if printall:
            print("z",str(layer+1),":",zl,sep="")
            print("a",str(layer+1),":",al,sep="")
        al_prev = al

    if debug:
        print ("==Finishing forwardProp.py")
    return za,al
