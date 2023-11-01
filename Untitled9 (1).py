#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
# Generate synthetic data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
# Define the learning rate and number of iterations
learning_rate = 0.01
num_iterations = 1000
# Initialize the slope (theta1) and intercept (theta0) with random values
theta0 = np.random.randn()
theta1 = np.random.randn()
# Perform gradient descent
for i in range(num_iterations):
 # Calculate the predictions
 y_pred = theta0 + theta1 * X
 
 # Calculate the error
 error = y_pred - y
 
 # Calculate the gradient with respect to theta0 and theta1
 gradient_theta0 = (1/len(X)) * np.sum(error)
 gradient_theta1 = (1/len(X)) * np.sum(error * X)
 
 # Update the parameters using the gradient and learning rate
 theta0 = theta0 - learning_rate * gradient_theta0
 theta1 = theta1 - learning_rate * gradient_theta1
# Print the final parameters
print("Intercept (theta0):", theta0)
print("Slope (theta1):", theta1)
# Plot the data and the regression line
plt.scatter(X, y, label='Data')
plt.plot(X, theta0 + theta1 * X, color='red', label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Gradient Descent')
plt.legend()
plt.show()


# In[ ]:




