# call as follows: 'images, labels ,images_test, labels_test = ld.loadData(debug=True)'
#

from mnist import MNIST
import numpy as np

def loadData(whichdataset="mnist", debug=False):
    # Loads a specified dataset into memory
    #       whichdataset - one of:
    #           "mnist" (digit images, public dataset)
    #           "fruit" (downloaded from fruit images from internet and converted to a file
    #   Returns: 
    #       array of same size [nx,m] where each input feature's mean = 0 and variance is 1

    #Some datasets won't contain test set
    images_test = None
    labels_test = None

    if debug:
        print ("==Starting loadData.py")

    if whichdataset=="mnist":
        mndata = MNIST('C:\labs\DeepLearnTest\mnist-data')
        images, labels = mndata.load_training()
        images_test, labels_test = mndata.load_testing()
    elif whichdataset=="fruit":
        images = np.load("C:\labs\FruitAugmentation\PicsToDatasetFile\images.npy") 
        labels = np.load("C:\labs\FruitAugmentation\PicsToDatasetFile\labels.npy") 
    else:
        raise Exception("loadData.py failed: unknown dataset")

    #images[_test] is a list of m elements, each of size nx. reshape to a matrix of size [nx X m]
    images = np.array(images,copy=False).T
    if images_test is not None:
        images_test = np.array(images_test,copy=False).T

    #labels[_test] is a list of m elements. reshape to numpy array of size [1 X m]
    labels=np.reshape(np.array(labels),(1,len(labels)))
    if labels_test is not None:
        labels_test=np.reshape(np.array(labels_test),(1,len(labels_test)))

    #sample to 1K training samples
    #images = images[:,0:1000]
    #labels = labels[:,0:1000]

    if debug:
        print ("images.shape:",images.shape)
        if images_test is not None:
            print ("images_test.shape:",images_test.shape)
        print ("labels.shape:",labels.shape)
        if labels_test is not None:
            print ("labels_test.shape:",labels_test.shape)
        print ("==Finishing loadData.py")

    return images, labels ,images_test, labels_test
 