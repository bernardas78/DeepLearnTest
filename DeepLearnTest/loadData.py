from mnist import MNIST
import numpy as np

# call as follows: 'images, labels ,images_test, labels_test = ld.loadData(debug=True)'

def loadData(debug=False):
    if debug:
        print ("==Starting loadData.py")

    mndata = MNIST('C:\labs\DeepLearnTest\mnist-data')

    images, labels = mndata.load_training()
    images_test, labels_test = mndata.load_testing()

    #images[_test] is a list of m elements, each of size nx. reshape to a matrix of size [nx X m]
    images = np.array(images,copy=False).T
    images_test = np.array(images_test,copy=False).T

    #labels[_test] is a list of m elements. reshape to numpy array of size [1 X m]
    labels=np.reshape(np.array(labels),(1,len(labels)))
    labels_test=np.reshape(np.array(labels_test),(1,len(labels_test)))

    #sample to 1K training samples
    #images = images[:,0:1000]
    #labels = labels[:,0:1000]

    if debug:
        print ("images.shape:",images.shape)
        print ("images_test.shape:",images_test.shape)
        print ("labels.shape:",labels.shape)
        print ("labels_test.shape:",labels_test.shape)
        print ("==Finishing loadData.py")

    return images, labels ,images_test, labels_test
 