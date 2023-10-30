import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Generate some example data for plotting
x_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)

# Get the parameters of the trained logistic regression model
w_out, b_out = lr_model.coef_[0], lr_model.intercept_[0]
print (w_out, b_out)

# Make a prediction using the trained model
x_pred = [[2, 1]]
y_pred = lr_model.predict(x_pred)
print ("Prediction:",y_pred)

# Plot the data points
plt.scatter(x_train[y_train == 0][:, 0], x_train[y_train == 0][:, 1], label="Class 0", marker="o")
plt.scatter(x_train[y_train == 1][:, 0], x_train[y_train == 1][:, 1], label="Class 1", marker="x")

# Plot the decision boundary
x_boundary = np.linspace(0, 3.5, 100)
y_boundary = -(w_out[0] * x_boundary + b_out) / w_out[1]
plt.plot(x_boundary, y_boundary, label="Decision Boundary")

# Plot the predicted point
plt.scatter(x_pred[0][0], x_pred[0][1], label=f"Prediction (Class {y_pred[0]})", marker="s", c='red')

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.title("Logistic Regression Decision Boundary and Prediction")
plt.show()