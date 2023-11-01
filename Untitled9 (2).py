#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
# Generate synthetic data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
# Create models for L1 and L2 regularization
lasso_model = Lasso(alpha=1.0) # L1 regularization (alpha parameter controls the strength)
ridge_model = Ridge(alpha=1.0) # L2 regularization (alpha parameter controls the strength)
# Fit the models to the data
lasso_model.fit(X, y)
ridge_model.fit(X, y)
# Print the coefficients and intercepts
print("Lasso Model Coefficients:", lasso_model.coef_)
print("Lasso Model Intercept:", lasso_model.intercept_)
print("Ridge Model Coefficients:", ridge_model.coef_)
print("Ridge Model Intercept:", ridge_model.intercept_)
# Plot the data and regression lines for Lasso and Ridge
plt.scatter(X, y, label='Data')
plt.plot(X, lasso_model.predict(X), color='red', label='Lasso Regression Line')
plt.plot(X, ridge_model.predict(X), color='green', label='Ridge Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('L1 and L2 Regularization for Linear Regression')
plt.legend()
plt.show()
# Calculate mean squared error for both models
lasso_mse = mean_squared_error(y, lasso_model.predict(X))
ridge_mse = mean_squared_error(y, ridge_model.predict(X))
print("Lasso Mean Squared Error:", lasso_mse)
print("Ridge Mean Squared Error:", ridge_mse)


# In[ ]:




