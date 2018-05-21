import matplotlib.pyplot as plt 
import numpy as np

def plotCosts (plotCache, costs, evaltest, coststest):
    # Plots costs of train (and, optionally, test) set(s)
    #
    #   plotCache: cache data returned from previous iterations 
    #   costs: array of train costs
    #   evaltest: indicator to draw/not test costs line
    #   coststest: array of test costs
    #   
    # Returns: plotCache dictionary with keys (used within next iterations):
    #   fig: the whole graph
    #   plt: subplot within
    #   mline [, mlinetest] - blue (and red) lines in the plot of train (and test) set cost

    #print ("==Starting plotCosts.py")
    if plotCache==None:
        plotCache = {}
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #print ("costs.shape:",costs.shape)
        mline, = ax.plot([], [], 'b-')
        if evaltest:
            mlinetest, = ax.plot([], [], 'r-')
        #store graph data for further iterations
        plotCache["fig"] = fig
        plotCache["ax"] = ax
        plotCache["mline"] = mline
        if evaltest:
            plotCache["mlinetest"] = mlinetest
    else:
        fig = plotCache["fig"]
        ax = plotCache["ax"]
        mline = plotCache["mline"]
        if evaltest:
            mlinetest = plotCache["mlinetest"]

    #Draw costs
    mline.set_xdata( np.linspace(1,len(costs),len(costs)) )
    mline.set_ydata(costs)
    if evaltest:
        mlinetest.set_xdata( np.linspace(1,len(coststest),len(coststest)) )
        mlinetest.set_ydata(coststest)
    ax.autoscale_view()
    ax.relim()
    fig.canvas.draw()
    fig.canvas.flush_events()

    #print ("==Finishing plotCosts.py")
    return plotCache