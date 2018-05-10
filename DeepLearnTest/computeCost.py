import numpy as np
import matplotlib.pyplot as plt

def computeCost(y, yhat, debug=False, printall=False):
    #   y: actual values (labels) of shape [n_y,m]
    #   yhat: predicted values (predictions) of shape [n_y,m]
    #Returns: Softmax cost: 1/m sum_all_samples ( sum_all_classes ( - y*log(yhat) ) )

    if debug:
        print ("==Starting computeCost.py")

    #for numerical stability, add epsilon to yhat when close to 0
    yhat[yhat<1e-200] = 1e-200

    loss = np.sum ( -np.multiply (y, np.log(yhat) ), axis=0, keepdims=True )

    if debug:
        print("loss.shape:",loss.shape)
        print("loss[:,1:5]:",loss[:,1:5])

    cost = np.mean(loss)

    if debug:
        print("cost:",cost)
        print ("==Finishing computeCost.py")
    return cost

