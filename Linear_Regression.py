import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([3,8,9,13,14,15,16,10,11,17])
Y = np.array([32,54,68,74,75,78,80,65,67,84])

print(X)

print(Y)

X = X.reshape(-1,1)

print(X)

model = LinearRegression()
model.fit(X,Y)

a = model.coef_[0]
b = model.intercept_


print("The value of a = (Slope Cofficient) : ", a)
print("The value of b = (Intercept) : ", b)

X_test = np.array([[12]])
Y_pred = model.predict(X_test)

print("Predict Salary for an employee with 12 years of experience : ", Y_pred)

#This is the part for evaluating the graph using scatter graph using linear regression

import matplotlib.pyplot as plt

plt.scatter(X,Y, color='blue', label='Data Value')

X_range = np.linspace(min(X), max(X), 100).reshape(-1,1)

Y_range_pred = model.predict(X_range)

plt.plot(X_range, Y_range_pred, color='red', linewidth=3, label='Best fit Regression Line')

plt.scatter(X_test, Y_pred, color = 'green', marker='X', s=100, label='Prediction for 12 years')

plt.xlabel('Years of experience')
plt.ylabel('Salary (in Thousands)')
plt.title('Salary Prediction using Linear Regression')
plt.legend()
plt.show()
