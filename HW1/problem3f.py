# Import necessary libraries
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Generate the data
m, n = 30, 100
A = np.random.randn(m, n)

# Generate sparse ground truth x0 (Bernoulli with rate 0.2)
x0 = np.random.binomial(1, 0.2, size=n)

# Generate noise epsilon ~ N(0, 0.5)
epsilon = np.random.normal(0, np.sqrt(0.5), size=m)

# Generate the output y = A @ x0 + epsilon
y = A @ x0 + epsilon

# Step 2: Define and solve the Lasso problem with CVXPY for different values of lambda
lambdas = [0.01, 0.1, 1.0]  # Try different values of lambda

# Initialize list to store solutions
x_stars = []

# Solve the Lasso problem for each lambda
for lam in lambdas:
    # Define the optimization variable
    x = cp.Variable(n)
    
    # Define the Lasso objective function
    objective = cp.Minimize((1/m) * cp.norm2(A @ x - y)**2 + lam * cp.norm1(x))
    
    # Solve the problem
    problem = cp.Problem(objective)
    problem.solve()
    
    # Store the solution
    x_stars.append(x.value)

# Step 3: Visualize the results
plt.figure(figsize=(10, 6))

# Plot the ground truth x0
plt.stem(x0, linefmt='g-', markerfmt='go', basefmt='r-', label='Ground truth x0')

# Plot the estimated x* for different lambdas
for i, lam in enumerate(lambdas):
    plt.stem(x_stars[i], linefmt=f'C{i+1}-', markerfmt=f'C{i+1}o', basefmt='r-', 
             label=f'Estimated x* for lambda={lam}')

# Set plot title and labels
plt.title('Lasso Regression: Ground Truth vs. Estimated x*')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.show()