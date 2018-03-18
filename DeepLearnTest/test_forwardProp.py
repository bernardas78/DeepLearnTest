X = np.array([[2.,3.,4.],[15.,1.,-1.]])
L = 2
layerdims = [2,3,1]
params = iw.initWeights(layerdims,"forRelu")
activations = ["relu","sigmoid"]
yhat = fp.forwardProp(L, params, activations, X, debug=True,printall=True)
