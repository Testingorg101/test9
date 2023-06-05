import numpy as np
from sklearn.linear_model import LinearRegression

# Sample input data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Predict values
X_test = np.array([[6], [7]])
y_pred = model.predict(X_test)

# Print the coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Print the predicted values
print("Predicted values:", y_pred)
