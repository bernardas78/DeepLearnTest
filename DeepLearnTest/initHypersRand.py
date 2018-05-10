import numpy as np

def initHypersRand(input_size):
    #Initializes hyperparameters in a random way
    #   input_size - #units in input layer
    #
    # Returns: dictionary of hyperparameters with keys:
    #   learning_rate
    #   L - number of layers (excl. input)
    #   activations - activation functions numpy array, size L (last always softmax)
    #   layerdims - dimensions of each layer, incl. input (size L+1). Last always 10 for digits
    #   optimization_technique - one of ["GradientDescent","GradientDescentWithMomentum","RMSProp","Adam"]
    #   beta_momentum - opt. techniques GradientDescentWithMomentum, Adam; optimization parameter beta1
    #   beta_rmsprop - opt. techniques RMSProp, Adam; optimization parameter beta2 (exp.weighted square averages)

    hypers = {}

    #make sure it's random
    np.random.seed()

    learning_rates = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-2]
    hypers["learning_rate"] = learning_rates[np.random.randint(0,len(learning_rates))]

    L_max = 4
    L = np.random.randint(2, L_max+1) #2 to L_max
    hypers["L"] = L

    layerdims_max = 256
    layerdims_min = 10
    layerdims_last = 10
    layerdims = np.random.randint(layerdims_min, layerdims_max+1, L-1)
    layerdims = np.append(input_size, layerdims)
    layerdims = np.append(layerdims, layerdims_last)
    hypers["layerdims"] = layerdims

    activations_last = "softmax"
    activations_all = np.array(["relu","tanh"])
    activations = activations_all[np.random.randint(0, len(activations_all), L-1)]
    activations = np.append(activations, activations_last)
    hypers["activations"] = activations

    #optimization_technique_all = np.array(["GradientDescent","GradientDescentWithMomentum","RMSProp","Adam"])
    optimization_technique_all = np.array(["RMSProp","Adam"])
    optimization_technique = optimization_technique_all[np.random.randint(0, len(optimization_technique_all))]
    hypers["optimization_technique"] = optimization_technique

    beta_momentum = 1 - np.power( 10., -2 * np.random.rand()) #1-10^(-r); r=[0;2]; beta1=[0;0.99]; median(beta1)=0.9
    hypers["beta_momentum"] = beta_momentum

    beta_rmsprop = 1 - np.power( 10., -6 * np.random.rand()) #1-10^(-r); r=[0;6]; beta2=[0;0.999999]; median(beta2)=0.999
    hypers["beta_rmsprop"] = beta_rmsprop

    return hypers