import numpy as np
import cvxpy as cp

np.random.seed(42)

n = 100
d = 20

### PART A ###

# Generate w* from N(0, 1)
w_star = np.random.normal(0, 1, d)

# Generate 100 iid normal vectors x_i in R^20
X = np.random.normal(0, 1, (n, d))

# Generate Laplace noise epsilon_i (mean 0, scale 1)
epsilon = np.random.laplace(0, 1, n)

# Calculate y_i = w*^T x_i + epsilon_i
y = X.dot(w_star) + epsilon

print("Generated Example:")
print("y_1 = ", y[0])
print("w* = ", w_star)
print("X_1 = ", X[0])
print("epsilon_1 = ", epsilon[0])

### PART B ###

#  Solving Least Squares (LS)
w_ls = cp.Variable(X.shape[1])
cost_ls = cp.sum_squares(X @ w_ls - y)
problem_ls = cp.Problem(cp.Minimize(cost_ls))
problem_ls.solve()

# Solving Least Absolute Deviations (LAD)
w_lad = cp.Variable(X.shape[1])
cost_lad = cp.norm1(X @ w_lad - y)
problem_lad = cp.Problem(cp.Minimize(cost_lad))
problem_lad.solve()

# Compute estimation errors
error_ls = np.linalg.norm(w_ls.value - w_star)
error_lad = np.linalg.norm(w_lad.value - w_star)

print()
print(f"Least Squares Estimation Error: {error_ls}")
print(f"Least Absolute Deviations Estimation Error: {error_lad}")

