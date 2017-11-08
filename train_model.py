from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
import read_data as loader

(titles_train, x_train, y_train), (titles_test, x_test, y_test) = loader.load_data()

# MACHINE LEARNING BITCH
model = LinearRegression()

# AAAAAAHHH
model.fit(x_train, y_train)

# TEST IT YO!
y_test_pred = model.predict(x_test)

# HOW IT BE?
# print('Accuracy:', accuracy_score(y_test, y_test_pred))
print('Mean absolute error:', mean_absolute_error(y_test, y_test_pred))