import numpy as np

def updateParams(L, params, grads, learning_rate, debug):
    #   L - layer count (excl.input layer)
    #   params: dictionary of weights, contains keys:
    #     W1,...,WL - where L number of layers
    #     b1,...,bL
    #   grads: gradients of params dictionary, contains keys:
    #     dW1,...,dWL 
    #     db1,...,dbL
    #   learing_rate:
    #   debug:
    #
    # Returns: updated dictionary of params:
    #     W1,...,WL - where L number of layers
    #     b1,...,bL

    if debug:
        print ("==Starting updateParams.py")

    for layer in range (0, L):
        W = params["W"+str(layer+1)]
        b = params["b"+str(layer+1)]
        dW = grads["dW"+str(layer+1)]
        db = grads["db"+str(layer+1)]

        if debug:
            indexes_5rows = np.random.randint(0,dW.shape[0],5)
            indexes_5cols = np.random.randint(0,dW.shape[1],5)

            print ("Rows, indexes:", indexes_5rows, indexes_5cols)
            print ("dW"+str(layer+1), ":", dW[indexes_5rows,indexes_5cols])
            print ("db"+str(layer+1), ":", db[indexes_5rows,0])
            print ("W"+str(layer+1)+"(before): ", W[indexes_5rows,indexes_5cols] )
            print ("b"+str(layer+1)+"(before): ", b[indexes_5rows,0] )

        W -= dW * learning_rate
        b -= db * learning_rate

        params["W"+str(layer+1)] = W
        params["b"+str(layer+1)] = b

        if debug:
            print ("W"+str(layer+1)+"(after): ", W[indexes_5rows,indexes_5cols] )
            print ("b"+str(layer+1)+"(after): ", b[indexes_5rows,0] )
    
    if debug:
        print ("==Finishing updateParams.py")

    return params