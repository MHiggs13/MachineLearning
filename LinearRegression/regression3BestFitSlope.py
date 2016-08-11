from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

"""
LESSON 8 - How to program the best fit slope
LESSON 9 - How to program the best fit line
LESSON 10- R Squared Theory - determining accuracy
-uses squared error, the error is the distance between the point and the line.
-this error is squared, so that only positive values are used
-error is also squared, so that outliers are penalized
-if you want to penalize for outliers you can use 4,, 6, 8 etc but 2 is
standard
-coefficient of determination aka r^2
-r^2 = 1- (SEy^ / SEmean(y)) # SE = Squared Error
# y^ = y hat...
LESSON 11- Programming R Squared
-Value of how good of a fit is the best fit line
"""
# data type not needed, just being explicit
# xs = np.array([1,2,3,4,5,6], dtype=np.float64)
# ys = np.array([5,4,6,5,6,7], dtype=np.float64)


"""
Returns a numpy array of the xs and a numpy array of the ys
hm: how many data points
variance: how variable do we want this data set to be
step: how far on average on average to step up the y value per point, assigned
    a default value
correlation: can pass a value to make correlation pos, neg or none
"""
def create_dataset(hm, variance, step=2, correlation=False):

    # first value for y
    val = 1
    # ys is an empty list
    ys = []

    # for however many (hm) points that need created
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)    # add y to ys

        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step

    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

def best_fit_slope_and_intercept(xs, ys):
    m = (((mean(xs) * mean(ys)) - mean(xs*ys))/
        ((mean(xs)**2) - mean(xs**2)))
    c = mean(ys) - m * mean(xs)
    return m, c

"""
The  amount of y distance is the error and this is squared to find the squared
error
"""
def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig)**2)

"""
coefficient of determindation = r**2, which is 1 minus the squared error of the
y hat line(the regression line) divided by the squared error of the mean of the
ys
"""
def coefficient_of_determination(ys_orig, ys_line):
    # makes the y mean line (a line with y mean value inplace of each original
    # y value
    y_mean_line = [mean(ys_orig) for y in ys_orig]

    # squared error of the regression line
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)

    return 1 - (squared_error_regr / squared_error_y_mean)

# create dataset using the create_dataset method
xs, ys = create_dataset(40, 80, 2, correlation='pos')

m, c  = best_fit_slope_and_intercept(xs, ys)
print('m is {}, c is {}'.format(m, c))

regression_line = [(m*x) + c for x in xs]

predict_x = 8
predict_y = (m*predict_x)+c # predict y where x = 8

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)


plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color='red')
plt.plot(xs, regression_line)
plt.show()
