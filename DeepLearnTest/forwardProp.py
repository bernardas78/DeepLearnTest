import numpy as np
import time

def forwardProp(L, params, activations, X, \
    regularization_technique="None", keep_prob=None, \
    debug=False, printall=False):
    #caclulates forward propagation of a neural network
    #   L - layer count (excl.input layer)
    #   params - dictionary containing "W1", "b1",..."WL", "bL" parameters
    #   activations - array of size L of activation functions, each of one of the values:
    #       "sigmoid"
    #       "tanh"
    #       "relu"
    #       "softmax"
    #   X - import matrix of shape [nx,m]
    #   regularization_technique: way to reduce overfitting to train set 
    #       one of: ["None","L2","Dropout"]
    #   keep_prob: vector of probabilities to keep node, size L (last value should be 1.); used in Dropout regularization
    #   debug:
    #   printall: 
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

        zl = np.add ( np.dot ( Wl, al_prev ), bl )
        #print ("a"+str(layer),":",np.mean(al_prev),np.max(al_prev),np.min(al_prev))
        #print ("W"+str(layer+1),":",np.mean(Wl),np.max(Wl),np.min(Wl))
        #print ("b"+str(layer+1),":",np.mean(bl),np.max(bl),np.min(bl))
        #print ("z"+str(layer+1),":",np.mean(zl),np.max(zl),np.min(zl))

        if activation == "sigmoid":
            al = 1. / ( 1. + np.exp (-zl))
        elif activation == "tanh":
            al = np.tanh ( zl )
        elif activation == "relu":
            al = np.fmax (zl, np.zeros(zl.shape, dtype="float64"))
        elif activation == "softmax":
            #for numerical stability, use "reduced" zl for exp
            zl_temp = zl - np.max(zl, axis=0)
            al = np.exp (zl_temp) / np.sum(np.exp (zl_temp), axis=0, keepdims=True)
        else:
            raise Exception("forwardProp.py failed: unknown activation")

        #Dropout regularization
        if regularization_technique=="None":
            pass
        elif regularization_technique=="L2":
            pass 
        elif regularization_technique=="Dropout":
            dl = np.random.rand(al.shape[0], al.shape[1]) < keep_prob[layer]
            al = np.multiply (al, dl) / keep_prob[layer]
        else:
            raise Exception("forwardProp.py failed: unknown regularization technique")


        za["Z"+str(layer+1)] = zl
        za["A"+str(layer+1)] = al

        if debug:
            print ("Z",str(layer+1),".shape:",zl.shape,"; A",str(layer+1),".shape:",al.shape,sep="")
            indexes_10hunits = np.random.randint(0,al.shape[0],10)
            indexes_10samples = np.random.randint(0,al.shape[1],10)
            print ("Z"+str(layer+1),"[", indexes_10hunits,",",indexes_10samples, "]: ", zl[indexes_10hunits, indexes_10samples], sep="" )
            print ("A"+str(layer+1),"[", indexes_10hunits,",",indexes_10samples, "]: ", al[indexes_10hunits, indexes_10samples], sep="" )

        if printall:
            print("z",str(layer+1),":",zl,sep="")
            print("a",str(layer+1),":",al,sep="")

        al_prev = al

    if debug:
        print ("==Finishing forwardProp.py")

    return za,al
