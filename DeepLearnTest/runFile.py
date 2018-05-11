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
n_x = images.shape[0]
n_y = 10

#reshape labels=[1,m] to y=[n_y,m]
y = np.zeros((n_y,m), dtype="float64")
y[labels, np.arange(m)] = 1.
#y[:,0:5]
#labels[0,0:5]


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
    
    yhat, grads, params = rm.runModel(iter_count=500, L=L, params=params, activations=activations,\
        X=images, y=y, learning_rate=learning_rate, minibach_size=1024,\
        optimization_technique=optimization_technique, beta_momentum=beta_momentum, beta_rmsprop=beta_rmsprop,\
        debug=debug, printcost=False)


#initialize NN architecture
#learning_rate=0.001
#L=2
#activations = ["relu","softmax"]
#layerdims = [n_x,100,n_y]
#optimization_technique="Adam"
#beta_momentum=0.9
#beta_rmsprop=0.999

    
learning_rate=0.01
L=4
activations = ["relu","relu","tanh","softmax"]
layerdims = [n_x,177,93,28,n_y]
optimization_technique="RMSProp"
beta_momentum = 0.580156929093833
beta_rmsprop = 0.999754127379942
params = iw.initWeights(layerdims,activations)
yhat, grads, params = rm.runModel(iter_count=100, L=L, params=params, activations=activations,\
    X=images, y=y, learning_rate=learning_rate, minibach_size=1024,\
    optimization_technique=optimization_technique, beta_momentum=beta_momentum, beta_rmsprop=beta_rmsprop,\
    debug=False, printcost=True)
cc.computeCost(y, yhat)
#yhat, grads, params = rm.runModel(iter_count=1000, L=L, params=params, activations=activations,\
#	X=images, y=y, learning_rate, minibach_size=1024,\
#	optimization_technique="RMSProp", beta_momentum=None, beta_rmsprop=0.999,\
#	debug=debug, printcost=True)

#yhat, grads, params = rm.runModel(iter_count=1000, L=L, params=params, activations=activations,\
#	X=images, y=y, learning_rate, minibach_size=1024,\
#	optimization_technique="GradientDescentWithMomentum", beta_momentum=0.9, beta_rmsprop=None,\
#	debug=debug, printcost=True)



===================installed libs
pip3 install matplotlib
pip3 install python-mnist
pip3 install sklearn
pip3 install scipy

