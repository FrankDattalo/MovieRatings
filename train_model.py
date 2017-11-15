# First, we will import the model that we would like to train.
from sklearn.linear_model import ElasticNet

# Next, we will import a class used for cross validation (hyper parameter selection)
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import PolynomialFeatures

# Then, we will import metrics to test our trained model
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Plotting library for data visualization
import matplotlib.pyplot as plt

# Finally, import our data loading functions 
import read_data as loader

loader.plot_histogram()

# This will load the data into respective training and testing sets
# the fist element is a list of titles where each title corresponds to the
# title of the example at the same index in x or y.
(titles_train, x_train, y_train), (titles_test, x_test, y_test) = loader.load_data()

# Here we will instatiate a model to train. 
# GridSearchCV will select the best model after exaustivly 
# trying all of the parameters within the map given as a parameter.
# Therefore, the map corresponds to the hyper parameters passed into the model.
# Each model will have different hyper parameters, so this map will need to 
# change when the model changes.
# n_jobs - the number of jobs to run in parallel
# verbose - debug printing 
model = GridSearchCV(ElasticNet(), 
                    {'l1_ratio': [0, .25, .5, .75, 1], 'alpha': [.01, .5, 1, 1.5, 2]},
                    verbose=2, n_jobs=16)

# Actual training is done within this method.
model.fit(x_train, y_train)

# Predicting test values here.
y_test_pred = model.predict(x_test)

# Defines a custom metric used to evaluate the model.
def accuracy_within_epsilon(y_test, y_pred, epsilon=.5):
    lower_bound = y_test - epsilon
    upper_bound = y_test + epsilon
    within_lower = (lower_bound <= y_pred).astype(int)
    within_higher = (y_pred <= upper_bound).astype(int)
    # same as doing a boolean and 1 * 1 = 1, 1 * 0 = 0 etc.
    within = within_lower * within_higher
    return sum(within) / len(within)

# Prints metrics of our model.
print()
print('Mean absolute error:', mean_absolute_error(y_test, y_test_pred))

print()
print('Mean squared error:', mean_squared_error(y_test, y_test_pred))

print()
print('Accuracy within 2.5:', accuracy_within_epsilon(y_test, y_test_pred, epsilon=2.5))
print('Accuracy within 2:  ', accuracy_within_epsilon(y_test, y_test_pred, epsilon=2))
print('Accuracy within 1.5:', accuracy_within_epsilon(y_test, y_test_pred, epsilon=1.5))
print('Accuracy within 1:  ', accuracy_within_epsilon(y_test, y_test_pred, epsilon=1))
print('Accuracy within .5: ', accuracy_within_epsilon(y_test, y_test_pred, epsilon=.5))
print('Accuracy within .25:', accuracy_within_epsilon(y_test, y_test_pred, epsilon=.25))
print('Accuracy within .1: ', accuracy_within_epsilon(y_test, y_test_pred, epsilon=.1))


plt.hist(y_test)
plt.hist(y_test_pred)
plt.title('IMDB score distribution between test and truth.')
plt.show()