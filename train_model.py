from sklearn.svm import SVR as Model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import read_data as loader

# loader.plot_histogram()

(titles_train, x_train, y_train), (titles_test, x_test, y_test) = loader.load_data()
model = GridSearchCV(Model(), 
                     {'kernel': ['rbf', 'sigmoid'], 'C': [.5, 1, 1.5]},
                     verbose=2)
model.fit(x_train, y_train)
y_test_pred = model.predict(x_test)

def accuracy_within_epsilon(y_test, y_pred, epsilon=.5):
    lower_bound = y_test - epsilon
    upper_bound = y_test + epsilon
    within_lower = (lower_bound <= y_pred).astype(int)
    within_higher = (y_pred <= upper_bound).astype(int)
    # same as doing a boolean and 1 * 1 = 1, 1 * 0 = 0 etc.
    within = within_lower * within_higher
    return sum(within) / len(within)

# HOW IT BE?
print('Mean absolute error:',   mean_absolute_error(y_test, y_test_pred))
print('Mean squared error:',    mean_squared_error(y_test, y_test_pred))
print('Accuracy within 2.5:',   accuracy_within_epsilon(y_test, y_test_pred, epsilon=2.5))
print('Accuracy within 2:',     accuracy_within_epsilon(y_test, y_test_pred, epsilon=2))
print('Accuracy within 1.5:',   accuracy_within_epsilon(y_test, y_test_pred, epsilon=1.5))
print('Accuracy within 1:',     accuracy_within_epsilon(y_test, y_test_pred, epsilon=1))
print('Accuracy within .5:',    accuracy_within_epsilon(y_test, y_test_pred, epsilon=.5))
print('Accuracy within .25:',   accuracy_within_epsilon(y_test, y_test_pred, epsilon=.25))

# plt.hist(y_test)
# plt.hist(y_test_pred)
# plt.show()