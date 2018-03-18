import importlib
import numpy as np
import gc
import matplotlib.pyplot as plt

#initial import. Calling 'exec(open("reimport.py").read())' from python command line will import these modules
# , but they will only be reloaded from files by calling re() command
import loadData as ld
import normalizeData as nd
import initWeights as iw

def re():
    importlib.reload(ld)
    importlib.reload(nd)
    importlib.reload(iw)
