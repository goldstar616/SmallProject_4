from __future__ import division
from __future__ import print_function

import numpy as np
import scipy as sp
import pandas as pd
import sklearn as sk
import matplotlib as mpl
import matplotlib.pylab as plt
import seaborn as sns
from SubwayDataHandler import SubwayDataReader
import statsmodels.api as sm
from statsmodels.graphics import utils
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
sns.set()

'''
Created on 2016. 1. 26.

@author: goldstar
'''

sdr = SubwayDataReader()
X, y = sdr.readData("data/sub.xlsx", "X1")
sdr.printData()

"""
df_seoul_station = pd.concat([X, y], axis=1)
df_seoul_station = sm.add_constant(df_seoul_station)
model_seoul_station = sm.OLS(df_seoul_station.ix[:, -1], df_seoul_station.ix[:, :-1])
result_seoul_station = model_seoul_station.fit()
print(result_seoul_station.summary())
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

cv = StratifiedKFold(y, n_folds=3, random_state=0)

for train_index, test_index in cv:
    print("test X:\n", X.ix[test_index].tail())
    print("." * 80 )        
    print("test y:", y.ix[test_index].tail())
    print("=" * 80 )

plt.figure(figsize=(15, 15))
sns.heatmap(np.corrcoef(X.T), annot=True, square=True, fmt='.2f', annot_kws={'size':15},
            yticklabels=X.columns, xticklabels=X.columns)
plt.show()