#exec(open("showPredicted.py").read())
import matplotlib.pyplot as plt
import numpy as np
import time


pred_cnt = 1000

#run model to get predictions (review: should use runModel.py)
_, yhat = fp.forwardProp(L, params, activations, X=images_test[:,0:pred_cnt], regularization_technique="None", keep_prob=None, debug=False)
pred = np.argmax(yhat, axis=0)

#prepare images for display
img = np.reshape(images_test[:,0:pred_cnt],(28,28,pred_cnt))
pred_correct = ytest[pred,range(pred_cnt)]==1
pred_color = np.repeat("green", pred_cnt)
pred_color[pred_correct != True] = "red"

rows=10
columns=10
 
plt.ion()
fig = plt.figure(figsize=(10, 5))
ims=[]
ims_lbl=[]

#create initial canvas with properties and some random data
for i in range(rows*columns):
    subplot = fig.add_subplot(rows,columns*2,i*2+1)
    subplot_lbl = fig.add_subplot(rows,columns*2,i*2+2)

    subplot.set_xticklabels([])
    subplot.set_yticklabels([])
    subplot.set_xticks([])
    subplot.set_yticks([])

    im = subplot.imshow(img[:,:,0], cmap='gray')
    ims = np.append(ims, im)

    subplot_lbl.set_xticklabels([])
    subplot_lbl.set_yticklabels([])
    subplot_lbl.set_xticks([])
    subplot_lbl.set_yticks([])

    im_lbl = subplot_lbl.text(x=0.5,y=0.4,s=1, ha="center", va="center", fontsize=20, color="red")
    ims_lbl = np.append(ims_lbl, im_lbl)

#refresh image a few times to show different subset
for j in range(0,9):
    for i in range(rows*columns):
        im = ims[i]
        im.set_data(img[:,:,(j*4)+i]*255)
        im_lbl = ims_lbl[i]
        im_lbl.set_text(pred[(j*4)+i])
        im_lbl.set_color(pred_color[(j*4)+i])
        #print("i,j,pred[(j*4)+i]:",i,j,pred[(j*4)+i])

    fig.canvas.flush_events()
