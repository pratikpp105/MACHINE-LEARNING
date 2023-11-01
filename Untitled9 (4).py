#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Import the required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Set a random seed for reproducibility
np.random.seed(0)

# Generate random data
X = 2 * np.random.rand(100, 1) - 1  # Random x values between -1 and 1
y = 2 * X**2 + 1 + np.random.randn(100, 1)  # Random quadratic function with noise

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose the degree of the polynomial
degree = 2  # You can change this to the desired polynomial degree

# Transform the features into polynomial features
poly_features = PolynomialFeatures(degree=degree)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# Fit a linear regression model to the polynomial features
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test_poly)

# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plot the true data and the predicted polynomial regression line
plt.scatter(X, y, label='True Data')
plt.plot(X, model.predict(poly_features.transform(X)), color='red', label=f'Polynomial Regression (Degree {degree})')
plt.legend()
plt.xlabel('X')
plt.ylabel('y')
plt.title(f'Polynomial Regression (Degree {degree})')
plt.show()


# In[ ]:




