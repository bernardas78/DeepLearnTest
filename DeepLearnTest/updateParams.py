import numpy as np

def updateParams(L, params, grads, learning_rate,\
    optimization_technique="GradientDescent",beta_momentum=None,beta_rmsprop=None,opt_params=None,\
    debug=False):
    #   L - layer count (excl.input layer)
    #   params: dictionary of weights, contains keys:
    #     W1,...,WL - where L number of layers
    #     b1,...,bL
    #   grads: gradients of params dictionary, contains keys:
    #     dW1,...,dWL 
    #     db1,...,dbL
    #   learing_rate:
    #   optimization_technique: way of how params (W's and b's) are updated
    #       one of ["GradientDescent","GradientDescentWithMomentum","RMSProp","Adam"]
    #   beta_momentum: opt. technique GradientDescentWithMomentum, Adam; optimization parameter beta1
    #   beta_rmsprop: opt. technique RMSProp, Adam; optimization parameter beta2 (exp.weighted square averages)
    #   opt_params: optimization parameters, dictionary for previously calculated values.
    #       GradientDescent - nothing
    #       GradientDescentWithMomentum - keys:
    #           dW[layer]_withMomentum_unbiased_prev - previously calulated dW'' with eliminated bias, for each layer 1:L
    #           db[layer]_withMomentum_unbiased_prev - .....................db''
    #       RMSProp - keys:
    #           SdW[layer] - previously calulated SdW, for each layer 1:L 
    #           Sdb[layer] -                      Sdb
    #       Adam:
    #   debug:
    #
    # Returns: updated dictionary of params:
    #     W1,...,WL - where L number of layers
    #     b1,...,bL
    
    #debug=True #SET TEMPORARILY!!

    epsilon = 1e-8
    if debug:
        print ("==Starting updateParams.py")

    for layer in range (0, L):
        W = params["W"+str(layer+1)]
        b = params["b"+str(layer+1)]
        dW = grads["dW"+str(layer+1)]
        db = grads["db"+str(layer+1)]

        if debug:
            indexes_4rows = np.random.randint(0,dW.shape[0],4)
            indexes_4cols = np.random.randint(0,dW.shape[1],4)
            #indexes_4rows = 0
            #indexes_4cols = 0

            print ("Rows, indexes:", indexes_4rows, indexes_4cols)
            print ("dW"+str(layer+1), ":", dW[indexes_4rows,indexes_4cols])
            print ("db"+str(layer+1), ":", db[indexes_4rows,0])
            print ("W"+str(layer+1)+"(before): ", W[indexes_4rows,indexes_4cols] )
            print ("b"+str(layer+1)+"(before): ", b[indexes_4rows,0] )
        
        if optimization_technique=="GradientDescent":
            W -= dW * learning_rate
            b -= db * learning_rate
        elif optimization_technique=="GradientDescentWithMomentum":
            if "dW"+str(layer+1)+"_withMomentum_unbiased_prev" in opt_params.keys():
                dW_withMomentum_unbiased_prev = opt_params["dW"+str(layer+1)+"_withMomentum_unbiased_prev"]
                db_withMomentum_unbiased_prev = opt_params["db"+str(layer+1)+"_withMomentum_unbiased_prev"]
                t = opt_params["t"]
            else:
                dW_withMomentum_unbiased_prev = np.zeros(dW.shape)
                db_withMomentum_unbiased_prev = np.zeros(db.shape)
                t = 1
            #dW_withMomentum = beta_momentum * dW_withMomentum_unbiased_prev + (1-beta_momentum) * dW
            #db_withMomentum = beta_momentum * db_withMomentum_unbiased_prev + (1-beta_momentum) * db
            #dW_withMomentum_unbiased = dW_withMomentum / (1-np.power(beta_momentum,t))
            #db_withMomentum_unbiased = db_withMomentum / (1-np.power(beta_momentum,t))
            dW_withMomentum_unbiased = dW_withMomentum_unbiased_prev * beta_momentum + (1-beta_momentum)*dW / (1-np.power(beta_momentum,t))
            db_withMomentum_unbiased = db_withMomentum_unbiased_prev * beta_momentum + (1-beta_momentum)*db / (1-np.power(beta_momentum,t))
            if debug:
                print ("dW"+str(layer+1)+"_withMomentum_unbiased_prev", ":", dW_withMomentum_unbiased_prev[indexes_4rows,indexes_4cols])
                print ("db"+str(layer+1)+"_withMomentum_unbiased_prev", ":", db_withMomentum_unbiased_prev[indexes_4rows,0])
                #print ("dW"+str(layer+1)+"_withMomentum", ":", dW_withMomentum[indexes_4rows,indexes_4cols])
                #print ("db"+str(layer+1)+"_withMomentum", ":", db_withMomentum[indexes_4rows,0])
                print ("dW"+str(layer+1)+"_withMomentum_unbiased", ":", dW_withMomentum_unbiased[indexes_4rows,indexes_4cols])
                print ("db"+str(layer+1)+"_withMomentum_unbiased", ":", db_withMomentum_unbiased[indexes_4rows,0])
                print ("t:",t)
            W -= dW_withMomentum_unbiased * learning_rate
            b -= db_withMomentum_unbiased * learning_rate
            #store calculated dW, db with momentum (and bias eliminated) for next iterations
            opt_params["dW"+str(layer+1)+"_withMomentum_unbiased_prev"] = dW_withMomentum_unbiased
            opt_params["db"+str(layer+1)+"_withMomentum_unbiased_prev"] = db_withMomentum_unbiased
        elif optimization_technique=="RMSProp":
            if "SdW"+str(layer+1) in opt_params.keys():
                SdW_prev = opt_params["SdW"+str(layer+1)]
                Sdb_prev = opt_params["Sdb"+str(layer+1)]
            else:
                SdW_prev = np.zeros(dW.shape)
                Sdb_prev = np.zeros(db.shape)
            SdW = beta_rmsprop * SdW_prev + (1-beta_rmsprop) * np.multiply (dW, dW)
            Sdb = beta_rmsprop * Sdb_prev + (1-beta_rmsprop) * np.multiply (db, db)
            if debug:
                print ("SdW"+str(layer+1), ":", SdW[indexes_4rows,indexes_4cols])
                print ("Sdb"+str(layer+1), ":", Sdb[indexes_4rows,0])
            W -= dW / (np.sqrt(SdW) + epsilon) * learning_rate
            b -= db / (np.sqrt(Sdb) + epsilon) * learning_rate
            #store calculated SdW, Sdb for next iterations
            opt_params["SdW"+str(layer+1)] = SdW
            opt_params["Sdb"+str(layer+1)] = Sdb
        elif optimization_technique=="Adam":
            if "VdW"+str(layer+1)+"_prev" in opt_params.keys():
                VdW_prev = opt_params["VdW"+str(layer+1)+"_prev"]
                Vdb_prev = opt_params["Vdb"+str(layer+1)+"_prev"]
                SdW_prev = opt_params["SdW"+str(layer+1)+"_prev"]
                Sdb_prev = opt_params["Sdb"+str(layer+1)+"_prev"]
                t = opt_params["t"]
            else:
                VdW_prev = np.zeros(dW.shape)
                Vdb_prev = np.zeros(db.shape)
                SdW_prev = np.zeros(dW.shape)
                Sdb_prev = np.zeros(db.shape)
                t = 1
            VdW = beta_momentum * VdW_prev + (1-beta_momentum) * dW
            Vdb = beta_momentum * Vdb_prev + (1-beta_momentum) * db
            VdW_unbiased = VdW / (1 - np.power(beta_momentum,t) )
            Vdb_unbiased = Vdb / (1 - np.power(beta_momentum,t) )
            SdW = beta_rmsprop * SdW_prev + (1-beta_rmsprop) * np.multiply (dW, dW)
            Sdb = beta_rmsprop * Sdb_prev + (1-beta_rmsprop) * np.multiply (db, db)
            SdW_unbiased = SdW / (1 - np.power(beta_rmsprop,t) )
            Sdb_unbiased = Sdb / (1 - np.power(beta_rmsprop,t) )
            if debug:
                print ("VdW"+str(layer+1)+"_prev", ":", VdW_prev[indexes_4rows,indexes_4cols])
                print ("Vdb"+str(layer+1)+"_prev", ":", Vdb_prev[indexes_4rows,0])
                print ("SdW"+str(layer+1)+"_prev", ":", SdW_prev[indexes_4rows,indexes_4cols])
                print ("Sdb"+str(layer+1)+"_prev", ":", Sdb_prev[indexes_4rows,0])
                print ("VdW"+str(layer+1), ":", VdW[indexes_4rows,indexes_4cols])
                print ("Vdb"+str(layer+1), ":", Vdb[indexes_4rows,0])
                print ("VdW"+str(layer+1)+"_unbiased", ":", VdW_unbiased[indexes_4rows,indexes_4cols])
                print ("Vdb"+str(layer+1)+"_unbiased", ":", Vdb_unbiased[indexes_4rows,0])
                print ("SdW"+str(layer+1), ":", SdW[indexes_4rows,indexes_4cols])
                print ("Sdb"+str(layer+1), ":", Sdb[indexes_4rows,0])
                print ("SdW"+str(layer+1)+"_unbiased", ":", SdW_unbiased[indexes_4rows,indexes_4cols])
                print ("Sdb"+str(layer+1)+"_unbiased", ":", Sdb_unbiased[indexes_4rows,0])
            W -= VdW_unbiased / (np.sqrt(SdW_unbiased) + epsilon) * learning_rate
            b -= Vdb_unbiased / (np.sqrt(Sdb_unbiased) + epsilon) * learning_rate
            #store calculated VdW, Vdb, SdW, Sdb (all unbiased) for next iterations
            opt_params["VdW"+str(layer+1)+"_prev"] = VdW
            opt_params["Vdb"+str(layer+1)+"_prev"] = Vdb
            opt_params["SdW"+str(layer+1)+"_prev"] = SdW
            opt_params["Sdb"+str(layer+1)+"_prev"] = Sdb
        else:
            raise Exception("updateParams.py failed: unknown optimization technique")


        params["W"+str(layer+1)] = W
        params["b"+str(layer+1)] = b

        if debug:
            print ("W"+str(layer+1)+"(after): ", W[indexes_4rows,indexes_4cols] )
            print ("b"+str(layer+1)+"(after): ", b[indexes_4rows,0] )
   
    #for next iteration - increase the t
    if optimization_technique in ("GradientDescentWithMomentum", "Adam"):
        opt_params["t"] = t+1 

    if debug:
        print ("==Finishing updateParams.py")

    return params, opt_params