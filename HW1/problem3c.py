import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

m = 100
n = 50
max_iter = 500
tolerance = 1e-6

A = np.random.randn(m, n)
y = np.random.randn(m)

# f(x) = ||y - Ax||_2^2
def objective(A, y, x):
    return np.linalg.norm(y - A @ x, 2)**2

# grad_f(x) = A^T(y-Ax)
def gradient(A, y, x):
    return A.T @ (A @ x - y)

# Backtracking line search
def backtracking_line_search(A, y, x, grad, alpha=0.4, beta=0.8):
    t = 1.0
    while objective(A, y, x - t * grad) > objective(A, y, x) - alpha * t * np.dot(grad, grad):
        t *= beta
    return t

# Gradient descent with backtracking line search
def gradient_descent(A, y, max_iter=500, tol=1e-6):
    x = np.zeros(n)  
    obj_values = []  
    
    for i in range(max_iter):
        grad = gradient(A, y, x)  # compute gradient
        obj_value = objective(A, y, x)  # current objective value
        obj_values.append(obj_value)
        
        if np.linalg.norm(grad) < tol:  # check convergence
            break
        
        # Compute step size using backtracking line search
        step_size = backtracking_line_search(A, y, x, grad)
        
        # Update x
        x = x - step_size * grad
    
    return x, obj_values

# Run gradient descent
x_opt, obj_values = gradient_descent(A, y)

# Plot the objective value vs iterations
plt.plot(obj_values)
plt.xlabel('Iteration')
plt.ylabel('Objective Value')
plt.title('Objective Value vs Iterations')
plt.grid(True)
plt.show()