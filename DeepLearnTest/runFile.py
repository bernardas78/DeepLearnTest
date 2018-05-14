cd c:\labs\DeepLearnTest\DeepLearnTest
python

exec(open("reimport.py").read())

debug=False

#read the data
images, labels, images_test, labels_test = ld.loadData(debug)

#normalize the data
images = nd.normalizeData(images, debug)
images_test = nd.normalizeData(images_test, debug)

m = images.shape[1]
mtest = images_test.shape[1]
n_x = images.shape[0]
n_y = 10

#reshape labels=[1,m] to y=[n_y,m]
y = np.zeros((n_y,m), dtype="float64")
y[labels, np.arange(m)] = 1.
ytest = np.zeros((n_y,mtest), dtype="float64")
ytest[labels_test, np.arange(mtest)] = 1.
#y[:,0:5]
#labels[0,0:5]

#Choosing the best model: randomly initialize hyperparams and architecture
for model in range(50):
    print ("Running model:" + str(model))
    
    #randomly initialize architecture and hyperparams
    hypers = ih.initHypersRand(input_size=n_x)
    learning_rate = hypers["learning_rate"]
    L = hypers["L"]
    activations = hypers["activations"]
    layerdims = hypers["layerdims"]
    optimization_technique = hypers["optimization_technique"]
    beta_momentum = hypers["beta_momentum"]
    beta_rmsprop = hypers["beta_rmsprop"]
    
    #initialize weights
    params = iw.initWeights(layerdims,activations)
    
    #train model
    iter_count=500
    params = tm.trainModel(iter_count=iter_count, L=L, params=params, activations=activations,\
        X=images, y=y, learning_rate=learning_rate, minibach_size=1024,\
        optimization_technique=optimization_technique, beta_momentum=beta_momentum, beta_rmsprop=beta_rmsprop,\
        debug=debug, drawcost=False, evaltest=False)
    
    #run model to get cost, accuracy
    _,costTrain, accuracyTrain = rm.runModel(L, params, activations, X=images, y=y, debug=False, printcost=False)
    
    #output hyper params and results to a file
    prtf.paramsResultsToFile(iter_count, L, params, activations, learning_rate,\
        optimization_technique,beta_momentum, beta_rmsprop, costTrain, accuracyTrain)



#initialize NN architecture
#learning_rate=0.001
#L=2
#activations = ["relu","softmax"]
#layerdims = [n_x,100,n_y]
#optimization_technique="Adam"
#beta_momentum=0.9
#beta_rmsprop=0.999

    
learning_rate=0.01
L=2
activations = ["tanh","softmax"]
layerdims = [n_x,239,n_y]
optimization_technique="Adam"
beta_momentum = 0.427799355026437
beta_rmsprop = 0.999462202362609
params = iw.initWeights(layerdims,activations)
params = tm.trainModel(iter_count=20, L=L, params=params, activations=activations,\
    X=images, y=y, learning_rate=learning_rate, minibach_size=1024,\
    optimization_technique=optimization_technique, beta_momentum=beta_momentum, beta_rmsprop=beta_rmsprop,\
    debug=False, drawcost=True, evaltest=True, Xtest=images_test, ytest=ytest)
print ("Train set results:")
_,_,_ = rm.runModel(L, params, activations, X=images, y=y, debug=False, printcost=True)
print ("Test set results:")
_,_,_ = rm.runModel(L, params, activations, X=images_test, y=ytest, debug=False, printcost=True)
#cc.computeCost(y, yhat)
#params = tm.trainModel(iter_count=1000, L=L, params=params, activations=activations,\
#	X=images, y=y, learning_rate, minibach_size=1024,\
#	optimization_technique="RMSProp", beta_momentum=None, beta_rmsprop=0.999,\
#	debug=debug, drawcost=True)

#params = tm.trainModel(iter_count=1000, L=L, params=params, activations=activations,\
#	X=images, y=y, learning_rate, minibach_size=1024,\
#	optimization_technique="GradientDescentWithMomentum", beta_momentum=0.9, beta_rmsprop=None,\
#	debug=debug, drawcost=True)



===================installed libs
pip3 install matplotlib
pip3 install python-mnist
pip3 install sklearn
pip3 install scipy

