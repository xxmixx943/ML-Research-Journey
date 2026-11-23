
import numpy as np

# Day 2: NumPy Batch Prediction

# Original dataset
X = np.array([
    [2, 8],
    [4, 7],
    [6, 6],
    [8, 5],
    [10, 4]
])

# Model parameters
w = np.array([5, 2])
b = 10

# Challenge 1: Batch prediction
print("Dataset shape:", X.shape)

predictions = X @ w + b
print("Original predictions:", predictions)

mean_pre = predictions.mean()
print("Mean prediction:", mean_pre)

# Challenge 2: Controlled experiment
X_new = X.copy()

# Increase study hours by 2
X_new[:, 0] += 2

# Decrease sleep hours by 1
X_new[:, 1] -= 1

new_predictions = X_new @ w + b

print("New predictions:", new_predictions)
print("Prediction changes:", new_predictions - predictions)
