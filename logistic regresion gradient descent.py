import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    w_history = []
    b_history = []
    w = w_in
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient_logistic(X, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        w_history.append(w.copy())
        b_history.append(b)

    return w, b, w_history, b_history

# Data
x_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

# Set the learning rate and number of iterations
w_tmp = np.zeros(x_train.shape[1])
b_tmp = 0
alpha = 0.1
iters = 10000

# Run gradient descent to train the logistic regression model and collect the history
w_out, b_out, w_history, b_history = gradient_descent(x_train, y_train, w_tmp, b_tmp, alpha, iters)

# Create a 3D plot of the cost function surface
w_min, w_max = np.min(w_history), np.max(w_history)
b_min, b_max = np.min(b_history), np.max(b_history)

w_range = np.linspace(w_min, w_max, 100)
b_range = np.linspace(b_min, b_max, 100)
W, B = np.meshgrid(w_range, b_range)
J = np.zeros_like(W)

for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        J[i, j] = compute_cost_logistic(x_train, y_train, np.array([W[i, j]]), B[i, j])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W, B, J, cmap='viridis')
ax.set_xlabel('W')
ax.set_ylabel('B')
ax.set_zlabel('Cost')
ax.set_title('Cost Function Surface')

# Plot the trajectory of gradient descent
ax.plot(w_history, b_history, [compute_cost_logistic(x_train, y_train, np.array([w]), b) for w, b in zip(w_history, b_history)], color='red', marker='o', label='Gradient Descent')

# Find the global minimum
global_min_index = np.unravel_index(np.argmin(J, axis=None), J.shape)
global_min_w = w_range[global_min_index[1]]
global_min_b = b_range[global_min_index[0]]
global_min_cost = J[global_min_index]

ax.plot([global_min_w], [global_min_b], [global_min_cost], color='green', marker='o', markersize=8, label='Global Minimum', alpha=1.0)
ax.legend()

plt.show()