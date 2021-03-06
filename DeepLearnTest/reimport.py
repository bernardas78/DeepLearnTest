import importlib
import numpy as np
import matplotlib.pyplot as plt

#initial import. Calling 'exec(open("reimport.py").read())' from python command line will import these modules
# , but they will only be reloaded from files by calling re() command
import loadData as ld
import normalizeData as nd
import initWeights as iw
import forwardProp as fp
import computeCost as cc
import backProp as bp
import updateParams as up
import trainModel as tm
import runModel as rm
import paramsResultsToFile as prtf
import initHypersRand as ih
import plotCosts as pc

def re():
    importlib.reload(ld)
    importlib.reload(nd)
    importlib.reload(iw)
    importlib.reload(fp)
    importlib.reload(cc)
    importlib.reload(bp)
    importlib.reload(up)
    importlib.reload(tm)
    importlib.reload(rm)
    importlib.reload(prtf)
    importlib.reload(ih)
    importlib.reload(pc)
