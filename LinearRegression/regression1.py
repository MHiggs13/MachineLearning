import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn import linear_model as lm
"""
Should check documentation of each algo being used.

For instance for linear regression, we check if it can be threaded. So we find 
n_jobs.

Linear regression can be threaded a "lot" compared to support vector regression

"""

# fetch data from quandl about company (df stands for data frame)
df = quandl.get('WIKI/GOOGL')
# filter data fetched to just the following titles
df = df[['Adj. Open', 'Adj. High', 'Adj. Low' , 'Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.00
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.00

df = df [['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace = True)

forecast_out = int(math.ceil(0.1*len(df))) # predicts "0.1" days in advance...
print(forecast_out) # 30

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace = True)

# features capital X, labels lowercase  y
X = np.array(df.drop(['label'], 1))	# drop returns a new data frame with just 'label'
y = np.array(df['label'])

"""
-scaling X before feeding it through the classifier
-o properly scale, it has to be scaled alongside the training values. this might
add to processing time. So in cases of say high frequency trading this step would 
probably be skipped
"""
X = preprocessing.scale(X)

# redefine x as a result of the above shift. making sure only have Xs where we have values for y
#X = X[:-forecast_out+1]  NOT NECESSARRY as the column that needs shifted is fropped above
y = np.array(df['label'])

# 20% of data is used as testing data (test_size)
# x train and y train used to fit our classifier
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# n_jobs being more should in theory decrease the time it takes to do the training
clf  = lm.LinearRegression(n_jobs=10)	# n_jobs = -1, as many jobs as processor cna handle
clf.fit(X_train, y_train)	# fit = train
accuracy = clf.score(X_test, y_test)	# score = test, could also be called confidence

# accuary of predicting the price shifted 1% of the days
print(accuracy)

# how easy it is to switch to a different algorithm
clf2 = svm.SVR(kernel='poly')	# default kernel is probably linear
clf2.fit(X_train, y_train)	# fit = train
accuracy2 = clf2.score(X_test, y_test)	# score = test, could also be called confidence

print(accuracy2)
