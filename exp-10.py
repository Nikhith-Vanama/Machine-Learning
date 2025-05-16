import numpy as np
import matplotlib.pyplot as plt
def locally_weighted_regression(x_query, X, y, tau):
    m = len(X)
    weights = np.exp(-np.square(X - x_query) / (2 * tau**2))  
    W = np.diag(weights)
    X_b = np.c_[np.ones((m, 1)), X]  
    theta = np.linalg.pinv(X_b.T @ W @ X_b) @ X_b.T @ W @ y
    return np.dot([1, x_query], theta)
np.random.seed(42)
X = np.linspace(0, 10, 100)  
y = np.sin(X) + np.random.normal(0, 0.1, X.shape[0])
tau = 1.0 
X_query = np.linspace(0, 10, 100)
y_pred = [locally_weighted_regression(x, X, y, tau) for x in X_query]
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label="Original Data", color="blue", alpha=0.5)
plt.plot(X_query, y_pred, label=f"LWR Fit (tau={tau})", color="red", linewidth=2)
plt.title("Locally Weighted Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()