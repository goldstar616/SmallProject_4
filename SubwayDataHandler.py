from __future__ import division
from __future__ import print_function

import numpy as np
import scipy as sp
import pandas as pd
import sklearn as sk
import matplotlib as mpl
import matplotlib.pylab as plt
import seaborn as sns
sns.set()

'''
Created on 2016. 1. 28.

@author: goldstar
'''

class SubwayDataReader(object):
    def __init__(self):
        self.X = None
        self.y = None
    
    def readData(self, path="", removeLabel="X1"):
        self.X = pd.read_excel(path, header=None)
        self.X.columns = ["X%d" % (i+1) for i in range(self.X.shape[1])]
        self.y = self.X[removeLabel]
        self.X = self.X.drop([removeLabel], axis=1)
        return self.X, self.y
    
    def printData(self):
        print(self.X.tail())
        print("--------------------------------------------")
        print(self.y.tail())
