# === PyTorch Workflow ===
# 1. Prepare and load data
# 2. Build a model
# 3. Fit the model to data (training)
# 4. Make predictions and evaluate the model (inference)
# 5. Save and load the model
# 6. Put everything together
# === Imports ===
import torch
from torch import nn 
import numpy
import matplotlib.pyplot as plt
import pathlib
from pathlib import Path
# === 1. Data Preparation ===

# In machine learning, the data can come in many forms. The key steps:
# 1. Convert the data into numerical representation
# 2. Build a model to learn patterns from the numerical data

# We'll create synthetic data using the linear regression formula: y = weight * x + bias
# Known parameters (ground truth):
weight, bias = 0.7, 0.3  # weight = slope, bias = y-intercept

# Generate input features (X) and corresponding labels (y)
start, end, step = 0, 1, 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)  # Shape: [N, 1]
y = weight * X + bias  # Linear relation

# Split the data into training and testing sets (80% train, 20% test)
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# === Visualization ===

def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test,
                     predictions=None):
    """
    Plot training data, test data, and model predictions (if provided).
    """
    plt.figure(figsize=(10,7))

    # Plot the training and test data
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training Data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing Data")

    # Plot predictions if available
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.xlabel("X - Input Feature")
    plt.ylabel("y - Target Value")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(prop={"size": 14})
    plt.title("Training vs Test Data Split", fontsize=16)

# === 2. Build the Model ===

# Model will:
# 1. Initialize with random values for weight and bias
# 2. Learn the correct values using gradient descent and backpropagation

# PyTorch Essentials:
# - nn.Module: Base class for all models
# - nn.Parameter: Parameters to learn
# - forward(): Defines the forward pass logic
# - torch.optim: Optimizers (e.g., SGD, Adam)

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

# === Initialize and Inspect the Model ===
torch.manual_seed(42)
model_0 = LinearRegressionModel()
print(model_0)
print(list(model_0.parameters()))
print(model_0.state_dict())

# === Inference Before Training ===
# Evaluate how the untrained model performs

with torch.inference_mode():
    y_pred = model_0(X_test)

plot_predictions(predictions=y_pred)
plt.savefig('plot.png')
plt.show()

# === 3. Train the Model ===

# Training = adjusting model weights from random to optimal using:
# - Loss function: Measures model error
# - Optimizer: Adjusts weights to minimize loss

# Define loss function and optimizer
loss_fun = nn.L1Loss()  # Mean Absolute Error
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

# Training loop steps:
# 1. Forward pass
# 2. Compute loss
# 3. Zero gradients
# 4. Backward pass
# 5. Update parameters

torch.manual_seed(42)
epochs = 300

train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    model_0.train()
    y_pred = model_0(X_train)
    loss = loss_fun(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Evaluate on test set
    model_0.eval()
    with torch.inference_mode():
        test_pred = model_0(X_test)
        test_loss = loss_fun(test_pred, y_test)

    if epoch % 10 == 0:
        epoch_count.append(epoch)
        train_loss_values.append(loss.detach().numpy())
        test_loss_values.append(test_loss.detach().numpy())
        print(f"Epoch: {epoch} | loss: {loss} | Test loss {test_loss}")
        print(model_0.state_dict())

# === Predictions After Training ===
with torch.inference_mode():
    y_preds_new = model_0(X_test)

plot_predictions(predictions=y_preds_new)
plt.savefig('plot_1.png')
plt.show()

# === Plot Training and Test Loss ===
plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and Test Loss Curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.savefig('plot_2.png')
plt.show()

# === 5. Save the Trained Model ===

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "O1_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
print(MODEL_SAVE_PATH)

# Save model state_dict (parameters)
torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)
