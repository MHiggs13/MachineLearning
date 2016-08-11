import pandas as pd
import quandl
import math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn import linear_model as lm
import matplotlib.pyplot as plt # used to plot stuff
from matplotlib import style # used to make plotting look decent
import pickle

print(plt.get_backend())

"""
REGRESSION - LESSONS 5, 6, _
Should check documentation of each algo being used.

For instance for linear regression, we check if it can be threaded. So we find 
n_jobs.

Linear regression can be threaded a "lot" compared to support vector regression

LESSON 6 - Pickling
Serialization of any python object, like a file save it and then can open it on
demand
Scaling - rent a cheap web server, put all the data on and prepare code to be
executed. Then upgrade to a more expensive server, run code. then save
classifier produced as a pickle. Then stop server.
"""
# select style to make plotting look decent
#style.use('ggplot') 

# fetch data from quandl about company (df stands for data frame)
df = quandl.get('WIKI/GOOGL')
# filter data fetched to just the following titles
df = df[['Adj. Open', 'Adj. High', 'Adj. Low' , 'Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.00
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.00

#           price        x             x            x
df = df [['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace = True)

forecast_out = int(math.ceil(0.1*len(df))) # predicts "0.1" days in advance...
print(forecast_out) # 30

df['label'] = df[forecast_col].shift(-forecast_out)


# features capital X, labels lowercase  y
X = np.array(df.drop(['label'], 1))	# drop returns a new data frame with just 'label'
"""
-scaling X before feeding it through the classifier
-o properly scale, it has to be scaled alongside the training values. this might
add to processing time. So in cases of say high frequency trading this step would 
probably be skipped
"""
X = preprocessing.scale(X)
# X_lately is what we are going to predict against,  find missing y from X_lately
X_lately = X[-forecast_out:] # last 30 days
X = X[:-forecast_out]


df.dropna(inplace = True)

# redefine x as a result of the above shift. making sure only have Xs where we have values for y
#X = X[:-forecast_out+1]  NOT NECESSARRY as the column that needs shifted is fropped above
y = np.array(df['label'])

# 20% of data is used as testing data (test_size)
# x train and y train used to fit our classifier
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# n_jobs being more should in theory decrease the time it takes to do the training
#clf  = lm.LinearRegression(n_jobs=10)	# n_jobs = -1, as many jobs as processor cna handle
#clf.fit(X_train, y_train)	# fit = train
"""
save classifier here, so that it can be reused on demand, might want to
retrain once a month or so, so that it stays up to date
"""
# save file as 'linearregression.pickle', wb indicates a wrtiing command, f is
# variable identifier
#with open('linearregression.pickle', 'wb') as f:
#    pickle.dump(clf, f)

# open a file ('linearregression.pickle'), stored as pickle_in, rb indicates a
# read 
pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)	# score = test, could also be called confidence

# accuary of predicting the price shifted 1% of the days
# print(accuracy)

# how easy it is to switch to a different algorithm
# clf2 = svm.SVR(kernel='poly')	# default kernel is probably linear
# clf2.fit(X_train, y_train)	# fit = train
# accuracy2 = clf2.score(X_test, y_test)	# score = test, could also be called confidence

# print(accuracy2)

# calculates the stock practices for the next 30 days
forecast_set = clf.predict(X_lately) # crux of doing a predicition with scikit

# prints next 30 days of stock pricesm accuracy, and the amount of prices printed
print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

"""
kind of having to do a work around as x are features and y are labels. So y just
so happens to match up to an axis on a graph. While x does not. So working out 
a day for each y value
"""
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400 # number of seconds in a day
next_unix = last_unix + one_day # sort of hardcoding here...

# populate data frame with new dates and forecast values
for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day # advance one day
	# setting day and y value into data frame, features one line for loop
	df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
        # for loc next_date is the index

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
