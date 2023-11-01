#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
# Generate synthetic data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
# Plot the data
plt.scatter(X, y, label='Data')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Example')
plt.show()
# Split the data into training and testing sets (if needed)
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Perform linear regression
from sklearn.linear_model import LinearRegression
# Create a Linear Regression model
model = LinearRegression()
# Fit the model to the data
model.fit(X, y)
# Print the coefficients (slope and intercept)
print("Slope (Coefficient):", model.coef_)
print("Intercept:", model.intercept_)
# Plot the regression line
plt.scatter(X, y, label='Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Example')
plt.legend()
plt.show()


# In[ ]:




