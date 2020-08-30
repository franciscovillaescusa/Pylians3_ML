import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import sys,os,time

##################################### INPUT ############################################
# data manager parameters
test_size    = 300
random_state = 1

# random forest parameters
n_estimators     = 1000
min_samples_leaf = 1
n_jobs           = -1
########################################################################################

# get the data: X has format [n_samples, n_features]
# Y is either a 1D vector with length n_samples, or a 2D with [n_samples, n_features]
data   = np.linspace(0, 10, 3000) #shape (3000,)
labels = data*3 + 2               #shape (3000,)
data = np.reshape(data, (-1,1))   #shape (3000,1)

# get the training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, \
                test_size=test_size, random_state=random_state)

# define the model and train
start = time.time()
model = RandomForestRegressor(n_estimators=n_estimators, 
                              min_samples_leaf=min_samples_leaf, n_jobs=n_jobs)
model.fit(X_train, y_train)
print('Time taken %.1f seconds'%(time.time()-start))

# compute errors on training and test sets
for X,Y in zip([X_train, X_test],[y_train, y_test]):

    # make predictions and compute root mean square error
    y_pred = model.predict(X)
    error = np.sqrt(np.mean((y_pred-Y)**2))
    print('rmse = %.3e'%error)

# get the feature importance. A 1D array that sums to 1.
importances = model.feature_importances_
