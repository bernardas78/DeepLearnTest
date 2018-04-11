import numpy as np
import matplotlib.pyplot as plt

def computeCost(y, yhat, debug=False, printall=False):
    #   y: actual values (labels) of shape [n_y,m]
    #   yhat: predicted values (predictions) of shape [n_y,m]
    #Returns: Softmax cost: 1/m sum_all_samples ( sum_all_classes ( - y*log(yhat) ) )
    m = y.shape[1]
    if debug:
        print("m:",m)

    loss = np.sum ( -np.multiply (y, np.log(yhat) ), axis=0, keepdims=True )

    assert (loss.shape == (1,m))
    if debug:
        print("loss:",loss.shape)
        print("loss[:,1:5]:",loss[:,1:5])
        plt.hist(loss[0,:],bins=50)
        plt.show()
    cost = np.mean(loss)

    return cost

