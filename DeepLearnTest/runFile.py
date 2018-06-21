cd c:\labs\DeepLearnTest\DeepLearnTest
python

exec(open("reimport.py").read())

debug=False

#read the data
images, labels, images_test, labels_test = ld.loadData(whichdataset="mnist",debug=debug)

#normalize the data
images = nd.normalizeData(images, debug)
if images_test is not None:
    images_test = nd.normalizeData(images_test, debug)

m = images.shape[1]
if images_test is not None:
    mtest = images_test.shape[1]

n_x = images.shape[0]
n_y = np.max(labels) + 1 #assume the largest value exists in training set

#reshape labels=[1,m] to y=[n_y,m]
y = np.zeros((n_y,m), dtype="float64")
y[labels, np.arange(m)] = 1.
if images_test is not None:
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
        regularization_technique="None", lambd=0.,\
        debug=debug, drawcost=False, evaltest=False)
    
    #run model to get cost, accuracy
    _,costTrain, accuracyTrain = rm.runModel(L, params, activations, X=images, y=y, debug=False, printcost=False)
    
    #output hyper params and results to a file
    prtf.paramsResultsToFile(iter_count, L, params, activations, learning_rate,\
        optimization_technique,beta_momentum, beta_rmsprop, 0., costTrain, accuracyTrain, 0., 0.)



#train and evaluate a single model with pre-defined hypers    
learning_rate=0.01
L=2
activations = ["tanh","softmax"]
layerdims = [n_x,239,n_y]
optimization_technique="Adam"
beta_momentum = 0.427799355026437
beta_rmsprop = 0.999462202362609
keep_prob = np.append( np.repeat(1.0, L-1), 1.)
params = iw.initWeights(layerdims,activations)
params = tm.trainModel(iter_count=200, L=L, params=params, activations=activations,\
    X=images, y=y, learning_rate=learning_rate, minibach_size=100,\
    optimization_technique=optimization_technique, beta_momentum=beta_momentum, beta_rmsprop=beta_rmsprop,\
    regularization_technique="Dropout", lambd=0., keep_prob=keep_prob,\
    debug=False, drawcost=True, evaltest=False, Xtest=images_test, ytest=ytest)
print ("Train set results:")
_,_,_ = rm.runModel(L, params, activations, X=images, y=y, debug=False, printcost=True)
print ("Test set results:")
_,_,_ = rm.runModel(L, params, activations, X=images_test, y=ytest, debug=False, printcost=True)


#Choosing the best lambda for L2 regularization:
learning_rate=0.01
L=2
activations = ["tanh","softmax"]
layerdims = [n_x,239,n_y]
optimization_technique="Adam"
beta_momentum = 0.427799355026437
beta_rmsprop = 0.999462202362609
regularization_technique="L2"
iter_count=500
for model in range(100):
    print ("Running model:" + str(model))  
    np.random.seed()
    lambd = np.random.rand() #lambd=[0;1]; median(lambd)=0.5
    params = iw.initWeights(layerdims,activations)
    params = tm.trainModel(iter_count=iter_count, L=L, params=params, activations=activations,\
        X=images, y=y, learning_rate=learning_rate, minibach_size=1024,\
        optimization_technique=optimization_technique, beta_momentum=beta_momentum, beta_rmsprop=beta_rmsprop,\
        regularization_technique=regularization_technique, lambd=lambd,\
        debug=False, drawcost=False, evaltest=True, Xtest=images_test, ytest=ytest)
    _,costTrain,accuracyTrain = rm.runModel(L, params, activations, X=images, y=y, debug=False, printcost=False)
    _,costTest,accuracyTest = rm.runModel(L, params, activations, X=images_test, y=ytest, debug=False, printcost=False)
    #output hyper params and results to a file
    prtf.paramsResultsToFile(iter_count, L, params, activations, learning_rate,\
        optimization_technique,beta_momentum, beta_rmsprop, regularization_technique, lambd, costTrain, accuracyTrain, costTest, accuracyTest)


#train and evaluate a single model L2 with deep network
learning_rate=0.01
L=7
activations = ["relu","tanh","relu","tanh","relu","tanh","softmax"]
layerdims = [n_x,600,500,400,300,200,100,n_y]
optimization_technique="GradientDescent"
beta_momentum = None
beta_rmsprop = None
params = iw.initWeights(layerdims,activations)
params = tm.trainModel(iter_count=500, L=L, params=params, activations=activations,\
    X=images, y=y, learning_rate=learning_rate, minibach_size=1024,\
    optimization_technique=optimization_technique, beta_momentum=beta_momentum, beta_rmsprop=beta_rmsprop,\
    regularization_technique="L2", lambd=0.,\
    debug=False, drawcost=True, evaltest=True, Xtest=images_test, ytest=ytest)
print ("Train set results:")
_,_,_ = rm.runModel(L, params, activations, X=images, y=y, debug=False, printcost=True)
print ("Test set results:")
_,_,_ = rm.runModel(L, params, activations, X=images_test, y=ytest, debug=False, printcost=True)


#Choosing the best keep_prob for Dropout regularization:
learning_rate=0.01
L=2
activations = ["tanh","softmax"]
layerdims = [n_x,239,n_y]
optimization_technique="Adam"
beta_momentum = 0.427799355026437
beta_rmsprop = 0.999462202362609
regularization_technique="Dropout"
iter_count=500
for model in range(50):
    print ("Running model:" + str(model))  
    np.random.seed()
    keep_prob = np.append ( np.repeat (np.random.rand(), L-1), 1.) #keep_prob=[0;1]; median(keep_prob)=0.5
    params = iw.initWeights(layerdims,activations)
    params = tm.trainModel(iter_count=iter_count, L=L, params=params, activations=activations,\
        X=images, y=y, learning_rate=learning_rate, minibach_size=1024,\
        optimization_technique=optimization_technique, beta_momentum=beta_momentum, beta_rmsprop=beta_rmsprop,\
        regularization_technique=regularization_technique, lambd=0., keep_prob=keep_prob,\
        debug=False, drawcost=False, evaltest=True, Xtest=images_test, ytest=ytest)
    _,costTrain,accuracyTrain = rm.runModel(L, params, activations, X=images, y=y, debug=False, printcost=False)
    _,costTest,accuracyTest = rm.runModel(L, params, activations, X=images_test, y=ytest, debug=False, printcost=False)
    #output hyper params and results to a file
    prtf.paramsResultsToFile(iter_count, L, params, activations, learning_rate,\
        optimization_technique,beta_momentum, beta_rmsprop, regularization_technique, lambd=0., keep_prob=keep_prob,\
        costTrain=costTrain, accuracyTrain=accuracyTrain, costTest=costTest, accuracyTest=accuracyTest)

#single model using Dropout
learning_rate=0.01
L=2
activations = ["tanh","softmax"]
layerdims = [n_x,239,n_y]
optimization_technique="Adam"
beta_momentum = 0.427799355026437
beta_rmsprop = 0.999462202362609
regularization_technique="Dropout"
iter_count=1000
keep_prob = np.append ( np.repeat (0.7, L-1), 1.) 
params = iw.initWeights(layerdims,activations)
params = tm.trainModel(iter_count=iter_count, L=L, params=params, activations=activations,\
    X=images, y=y, learning_rate=learning_rate, minibach_size=1024,\
    optimization_technique=optimization_technique, beta_momentum=beta_momentum, beta_rmsprop=beta_rmsprop,\
    regularization_technique=regularization_technique, lambd=0., keep_prob=keep_prob,\
    debug=False, drawcost=True, evaltest=False, Xtest=images_test, ytest=ytest)
_,costTrain,accuracyTrain = rm.runModel(L, params, activations, X=images, y=y, debug=False, printcost=True)
_,costTest,accuracyTest = rm.runModel(L, params, activations, X=images_test, y=ytest, debug=False, printcost=True)


#train and evaluate a single model L2 with deep network
learning_rate=0.01
L=2
activations = ["tanh","softmax"]
layerdims = [n_x,239,n_y]
optimization_technique="Adam"
beta_momentum = 0.427799355026437
beta_rmsprop = 0.999462202362609
iter_count=100
params = iw.initWeights(layerdims,activations)
params = tm.trainModel(iter_count=iter_count, L=L, params=params, activations=activations,\
    X=images, y=y, learning_rate=learning_rate, minibach_size=1024,\
    optimization_technique=optimization_technique, beta_momentum=beta_momentum, beta_rmsprop=beta_rmsprop,\
    regularization_technique="L2", lambd=2.0,\
    debug=False, drawcost=True, evaltest=False, Xtest=images_test, ytest=ytest,\
    evalModel=False, evalModelFreqIter=20)

#train deeper net using 3 Dropout and 3 L2 and 1 no-reg model
learning_rate=0.001
L=4
activations = ["relu","tanh","tanh","softmax"]
layerdims = [n_x,251,159,235,n_y]
optimization_technique="Adam"
beta_momentum = 0.55439934667339
beta_rmsprop = 0.999989408193322
iter_count=5000
regularization_technique = ["None", "L2", "L2", "L2", "Dropout", "Dropout", "Dropout"]
lambd = [None, .5, 1., 2., None, None, None]
keep_prob = [None, None, None, None, [.8,.8,.8,1.], [.7,.7,.7,1.], [.6,.6,.6,1.]]
for model in range (len(regularization_technique)):
    params = iw.initWeights(layerdims,activations)
    params = tm.trainModel(iter_count=iter_count, L=L, params=params, activations=activations,\
        X=images, y=y, learning_rate=learning_rate, minibach_size=1024,\
        optimization_technique=optimization_technique, beta_momentum=beta_momentum, beta_rmsprop=beta_rmsprop,\
        regularization_technique=regularization_technique[model], lambd=lambd[model], keep_prob=keep_prob[model],\
        debug=False, drawcost=False, evaltest=True, Xtest=images_test, ytest=ytest,\
        evalModel=True, evalModelFreqIter=20)



===================installed libs
pip3 install matplotlib
pip3 install python-mnist
pip3 install sklearn
pip3 install scipy

