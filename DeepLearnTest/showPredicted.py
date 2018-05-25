import matplotlib.pyplot as plt
import numpy as np
import time
 
img = np.reshape(images[:,0:1000],(28,28,1000))

rows=10
columns=10
 
plt.ion()
fig = plt.figure(figsize=(5, 5))
ims=[]

for i in range(rows*columns):
    subplot = fig.add_subplot(rows,columns,i+1)

    subplot.set_xticklabels([])
    subplot.set_yticklabels([])
    subplot.set_xticks([])
    subplot.set_yticks([])

    im = subplot.imshow(img[:,:,0], cmap='gray')
    ims = np.append(ims, im)

for j in range(0,10):
    for i in range(rows*columns):
        im = ims[i]
        im.set_data(img[:,:,(j*4)+i]*255)

    #im[rows*columns-1].set_data(
    fig.canvas.flush_events()
