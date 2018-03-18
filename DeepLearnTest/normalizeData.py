import numpy as np
import sklearn.preprocessing
import matplotlib.pyplot as plt

def normalizeData(images, images_test, debug=False):
    #data is a numpy array of size [nx X m]
    #   it sets each input feature's mean to 0 and variance to 1
    # np.std() throws MemoryError for big data sets, therefore, divide in batches in order to normalize

    images_scaled = sklearn.preprocessing.scale(images, axis=1, copy=False)
    images_test_scaled = sklearn.preprocessing.scale(images_test, axis=1, copy=False)

    del images, images_test

    return images_scaled, images_test_scaled

    # m - number of training examples
    # nx - number of features
    # means - average values of each feature of shape [nx X 1]
    # variance_cum - variance (cumulatively gains its value) of shape [nx X 1]
    #                standard deviation formula: sqrt(sum(x-xmean)/m)


    #m = data.shape[1]
    #nx = data.shape[0]
    #means = np.mean(data, axis=1, keepdims=True)
    #variance_cum = np.zeros((nx,1))

    #for blockstart in range ( 0, m, blockSize):
        
    #    if blockstart + blockSize > m:
    #        blockend = m
    #    else:
    #        blockend = blockstart + blockSize
    #    data_block = data [ : , blockstart:blockend ]
    #    variance_cum += np.sum ( np.square ( data_block - means ), axis=1, keepdims=True) / m
    #if debug:
    #    print ("data.shape,means.shape,variance_cum.shape:",data.shape,means.shape,variance_cum.shape)
    #data = np.subtract(data,means)
    #data = np.divide (data, np.sqrt(variance_cum) )
    #return data

    #bellow throws MemoryError for dataset 60000x784
    #return np.divide(data - np.mean(data, axis=1, keepdims=True), np.std(data, axis=1, keepdims=True))

