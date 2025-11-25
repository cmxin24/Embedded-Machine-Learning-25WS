import numpy as np

# Exercise 3.1: Calculating Gradients
print("="*60)
print("Exercise 3.1: Calculating Gradients")
print("="*60)

# Define the network weights and biases
W1 = np.array([
    [1, -1, 0.5],
    [0.5, 1, 1],
    [1, -1, 0.5],
    [0.5, -1, 0],
    [1, -3, 1]
])

b1 = np.array([1, 2, 3])

W2 = np.array([
    [1],
    [0.5],
    [-0.5]
])

b2 = np.array([0])

# Input and target
X = np.array([1, -1, 0, -2, 2])
Y = np.array([-1.75])

print("\nInitial Weights:")
print("W1 (5x3):\n", W1)
print("\nb1:", b1)
print("\nW2 (3x1):\n", W2)
print("\nb2:", b2)
print("\nInput X:", X)
print("Target Y:", Y)

# Step 1: Forward pass
print("\n" + "="*40)
print("FORWARD PASS")
print("="*40)

# Layer 1: Linear transformation
z1 = X @ W1 + b1
print("\nLayer 1 (before activation):")
print("z1 = X @ W1 + b1 =", z1)

# Apply ReLU activation
h1 = np.maximum(0, z1)
print("\nAfter ReLU activation:")
print("h1 = ReLU(z1) =", h1)

# Layer 2: Linear transformation (output layer)
z2 = h1 @ W2 + b2
print("\nOutput layer:")
print("z2 = h1 @ W2 + b2 =", z2)

# Network output (no activation on output layer for regression)
y_pred = z2
print("\nNetwork output: y_pred =", y_pred[0])

# Step 2: Calculate Loss (MSE)
print("\n" + "="*40)
print("LOSS CALCULATION")
print("="*40)

n = 1  # Single sample
loss = 0.5 * (Y[0] - y_pred[0])**2 / n
print(f"\nMSE Loss = 1/(2n) * (Y - y_pred)² = 1/2 * ({Y[0]} - {y_pred[0]})²")
print(f"Loss = {loss}")

# Step 3: Backward pass - Calculate gradients
print("\n" + "="*40)
print("BACKWARD PASS - GRADIENT CALCULATION")
print("="*40)

# Gradient of loss with respect to output
dL_dy = -(Y[0] - y_pred[0]) / n
print(f"\n∂L/∂y_pred = -(Y - y_pred)/n = -({Y[0]} - {y_pred[0]})/1 = {dL_dy}")

# Gradients for Layer 2
dL_dW2 = h1.reshape(-1, 1) * dL_dy
dL_db2 = dL_dy

print("\nLayer 2 gradients:")
print("∂L/∂W2 = h1ᵀ * ∂L/∂y_pred:")
print(dL_dW2)
print(f"\n∂L/∂b2 = ∂L/∂y_pred = {dL_db2}")

# Backpropagate to hidden layer
dL_dh1 = dL_dy * W2.flatten()
print("\n∂L/∂h1 = ∂L/∂y_pred * W2ᵀ =", dL_dh1)

# Gradients through ReLU
dL_dz1 = dL_dh1 * (z1 > 0).astype(float)
print("\n∂L/∂z1 (through ReLU) =", dL_dz1)

# Gradients for Layer 1
dL_dW1 = np.outer(X, dL_dz1)
dL_db1 = dL_dz1

print("\nLayer 1 gradients:")
print("∂L/∂W1 = Xᵀ * ∂L/∂z1:")
print(dL_dW1)
print("\n∂L/∂b1 = ∂L/∂z1 =", dL_db1)

# Step 4: Update weights with learning rate λ = 0.1
print("\n" + "="*40)
print("WEIGHT UPDATE (λ = 0.1)")
print("="*40)

learning_rate = 0.1

# Update weights and biases
W1_new = W1 - learning_rate * dL_dW1
b1_new = b1 - learning_rate * dL_db1
W2_new = W2 - learning_rate * dL_dW2
b2_new = b2 - learning_rate * dL_db2

print("\nUpdated W1:")
print(W1_new)
print("\nUpdated b1:", b1_new)
print("\nUpdated W2:")
print(W2_new)
print("\nUpdated b2:", b2_new)

# Verify with new forward pass
z1_new = X @ W1_new + b1_new
h1_new = np.maximum(0, z1_new)
y_pred_new = h1_new @ W2_new + b2_new
loss_new = 0.5 * (Y[0] - y_pred_new[0])**2

print("\n" + "="*40)
print("VERIFICATION")
print("="*40)
print(f"Original loss: {loss}")
print(f"New prediction: {y_pred_new[0]}")
print(f"New loss: {loss_new}")
print(f"Loss decreased: {loss > loss_new}")
