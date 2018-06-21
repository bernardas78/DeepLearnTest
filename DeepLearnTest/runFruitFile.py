cd c:\labs\DeepLearnTest\DeepLearnTest
python

exec(open("reimport.py").read())

debug=False

#read the data
images, labels, images_test, labels_test = ld.loadData(whichdataset="fruit",debug=debug)

#normalize the data
images = nd.normalizeData(images, debug)
#if images_test is not None:
#    images_test = nd.normalizeData(images_test, debug)

m = images.shape[1]
#if images_test is not None:
#    mtest = images_test.shape[1]

n_x = images.shape[0]
n_y = np.max(labels) + 1 #assume the largest value exists in training set

#reshape labels=[1,m] to y=[n_y,m]
y = np.zeros((n_y,m), dtype="float64")
y[labels, np.arange(m)] = 1.
#if images_test is not None:
#    ytest = np.zeros((n_y,mtest), dtype="float64")
#    ytest[labels_test, np.arange(mtest)] = 1.
#y[:,0:5]
#labels[0,0:5]

#perform random permutation of data
dataorder = np.random.permutation(np.arange(m))
images = images[:,dataorder]
labels = labels[:,dataorder]

#Best alg on mnist gave 43% accuracy after 500 iterations. Need a deeper network...
learning_rate=0.01
activations = ["tanh","softmax"]
L=len(activations)
layerdims = [n_x,239,n_y]
optimization_technique="Adam"
beta_momentum = 0.427799355026437
beta_rmsprop = 0.999462202362609
keep_prob = None
params = iw.initWeights(layerdims,activations)
params = tm.trainModel(iter_count=500, L=L, params=params, activations=activations,\
    X=images[:,0:10000], y=y[:,0:10000], learning_rate=learning_rate, minibach_size=200,\
    optimization_technique=optimization_technique, beta_momentum=beta_momentum, beta_rmsprop=beta_rmsprop,\
    regularization_technique="None", lambd=0., keep_prob=keep_prob,\
    debug=False, drawcost=True, evaltest=False, Xtest=None, ytest=None)
print ("Train set results:")
_,_,_ = rm.runModel(L, params, activations, X=images[:,0:10000], y=y[:,0:10000], debug=False, printcost=True)
_,_,_ = rm.runModel(L, params, activations, X=images, y=y, debug=False, printcost=True)

#Trying a deeper network: accuracy 50% (500 iter) and 51% (2000 iter)
learning_rate=0.00001
activations = ["relu","tanh","tanh","softmax"]
L=len(activations)
layerdims = [n_x,251,159,235,n_y]
optimization_technique="Adam"
beta_momentum = 0.9
beta_rmsprop = 0.999
keep_prob = None
params = iw.initWeights(layerdims,activations)
params = tm.trainModel(iter_count=100, L=L, params=params, activations=activations,\
    X=images[:,20000:30000], y=y[:,20000:30000], learning_rate=learning_rate, minibach_size=200,\
    optimization_technique=optimization_technique, beta_momentum=beta_momentum, beta_rmsprop=beta_rmsprop,\
    regularization_technique="None", lambd=0., keep_prob=keep_prob,\
    debug=False, drawcost=True, evaltest=False, Xtest=None, ytest=None)
print ("Train set results:")
_,_,_ = rm.runModel(L, params, activations, X=images[:,20000:30000], y=y[:,20000:30000], debug=False, printcost=True)
