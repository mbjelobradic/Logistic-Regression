import numpy as np
import copy
import matplotlib.pyplot as plt

# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Compute the logistic cost
def compute_cost_logistic(X, y, w, b):
    m = X.shape[0]
    cost = -np.mean(y * np.log(sigmoid(np.dot(X, w) + b)) + (1 - y) * np.log(1 - sigmoid(np.dot(X, w) + b)))
    return cost

# Compute the gradient of the logistic cost
def compute_gradient_logistic(X, y, w, b):
    m = X.shape[0]
    f_wb = sigmoid(np.dot(X, w) + b)
    dj_dw = np.dot(X.T, (f_wb - y)) / m
    dj_db = np.sum(f_wb - y) / m
    return dj_dw, dj_db

# Perform gradient descent
def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    J_history = []  # Track the cost function's value over iterations
    w = copy.deepcopy(w_in)  # Create a copy of initial weights to avoid modifying the original
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient_logistic(X, y, w, b)
        w -= alpha * dj_dw  # Update weights using the gradient and learning rate
        b -= alpha * dj_db  # Update bias using the gradient and learning rate

        J_history.append(compute_cost_logistic(X, y, w, b))  # Store the cost for this iteration

        if i % (num_iters // 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}")  # Print the cost periodically

    return w, b, J_history

# Generate some example data for plotting
x_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

# Set the learning rate and number of iterations
w_tmp = np.zeros(x_train.shape[1])  # Initial weights
b_tmp = 0  # Initial bias
alpha = 0.1  # Learning rate
iters = 10000  # Number of gradient descent iterations

# Run gradient descent to train the logistic regression model and collect the updated weights and bias
w_out, b_out, _ = gradient_descent(x_train, y_train, w_tmp, b_tmp, alpha, iters)
print(f"Updated parameters: w = {w_out}, b = {b_out}")

# Plot the data points
plt.scatter(x_train[y_train == 0][:, 0], x_train[y_train == 0][:, 1], label="Class 0", marker="o")
plt.scatter(x_train[y_train == 1][:, 0], x_train[y_train == 1][:, 1], label="Class 1", marker="x")

# Plot the decision boundary
x_boundary = np.linspace(0, 3.5, 100)
y_boundary = -(w_out[0] * x_boundary + b_out) / w_out[1]
plt.plot(x_boundary, y_boundary, label="Decision Boundary")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.title("Logistic Regression Decision Boundary")
plt.show()