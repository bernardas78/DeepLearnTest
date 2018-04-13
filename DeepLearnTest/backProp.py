import numpy as np

def backProp(L, activations, params, za, y, debug=False):
    #   L - layer count (excl.input layer)
    #   activations - list of activation functions, left to right, e.g. ["relu","relu","softmax"]
    #   params - dictionary containing keys:
    #       W1, ..., WL - where L number of layers
    #       b1, ..., bL
    #   za - dictionary of intermiediate values (Z's and A'ctivations); contains keys:
    #       A0, ... AL - where L number of layers
    #       Z1, ... ZL
    #   y - actual values of size [n_y, m]
    #Returns: grads (dictionary) which contains keys:
    #           dW1, .., dWL - where L number of layers
    #           db1, .., dbL

    grads = {}
    m = y.shape[1]

    #dA = np.divide(-y,A)
    #n_y = y.shape[0]

    for layer in np.arange(start=L-1,stop=-1, step=-1):     # L-1,..,0
        activation = activations[layer]
        A = za["A"+str(layer+1)]
        Z = za["Z"+str(layer+1)]
        Aprev = za["A"+str(layer)]
        W = params["W"+str(layer+1)]
        nout = W.shape[0]

        if activation=="softmax":
                                                                      # Every matrix element indexes:
            x_ind = np.tile ( np.tile ( np.arange ( nout ), nout), m)   #    x'th row 01..901..9..
            y_ind = np.tile ( np.repeat ( np.arange ( nout ), nout), m) #    y'th column 00..11 ... 00..11
            z_ind = np.repeat ( np.arange ( m ), nout*nout )            #    z'th training example 0000..1111..60000
                                                                      # Diagonal indexes:
            xy_ind = np.tile ( np.arange(nout), m)                     #    xy'th row, xy'th column 01..901..
            zz_ind = np.repeat( np.arange(m), nout)                    #    zz'th training example 0000..1111..600000
            #derivative of AL with respect to ZL of shape [nout,nout,m]
            #dAL_dZL = np.zeros((nout,nout,m),dtype="float64")
            #dAL_dZL[x_ind,y_ind,z_ind] = - np.multiply( AL[x_ind,z_ind], AL[y_ind,z_ind] )  # =-Ai*Aj - when i<>j
            #dAL_dZL[xy_ind,xy_ind,zz_ind] = \
            #    np.multiply( AL[xy_ind,z_ind], (np.subtract(1.,AL[xy_ind,zz_ind]) ) )       # =Ai*(1-Ai) - when i==j
            #dZL = np.multiply( np.reshape(dAL,(1,nout,m)) , dAL_dZL)                         
            #dZL = np.sum(dZL, axis=1)
            dZ = np.zeros((nout,nout,m),dtype="float64")
            dZ[x_ind,y_ind,z_ind] = np.multiply ( y[y_ind,z_ind], A[x_ind,z_ind] )                       # =yi*Aj - when i<>j
            dZ[xy_ind,xy_ind,zz_ind] = np.multiply ( y[xy_ind,zz_ind], np.subtract(A[xy_ind,zz_ind],1) ) # =yi*(Ai-1) - when i==j
            dZ = np.sum(dZ, axis=1)                                                                      # dZ2=y1*A2+y2*(A2-1)+y3*a2+...++yC*a2
        elif activation=="relu":
            dZ = np.zeros((nout,m),dtype="float64")
            dZ[Z>0]=1.
        else:
            raise Exception("backProp.py failed: unknown activation")
        
        dW = 1./m * np.dot( dZ, Aprev.T)
        db = 1./m * np.sum(dZ, axis=1, keepdims=True)
        dA = np.dot (W.T, dZ)
 
        grads["dW"+str(layer+1)] = dW
        grads["db"+str(layer+1)] = db
 
    return grads