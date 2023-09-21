import torch
import pandas as pd
import matplotlib.pyplot as plt

# Read data into a pandas dataframe
data = pd.read_csv('day_head_circumference.csv')
x_train = torch.tensor(data['day'].values, dtype=torch.float32).reshape(-1, 1)
y_train = torch.tensor(data['head_circumference'].values, dtype=torch.float32).reshape(-1, 1)

mean = x_train.mean()
std = x_train.std()
x_train_normalized = (x_train - mean) / std

class NonLinearRegressionModel:

    def __init__(self):
        # Model variables
        self.W = torch.randn([1, 1], requires_grad=True)
        self.b = torch.randn([1, 1], requires_grad=True)

        # Predictor

    def f(self, x):
        return 20 * torch.sigmoid(x @ self.W + self.b) + 31

        # Loss function

    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))


model = NonLinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], 0.00001)
for epoch in range(1000000):
    model.loss(x_train_normalized, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
sorted_indices = torch.argsort(x_train, dim=0).squeeze()
x_train_sorted = x_train[sorted_indices]
predictions_sorted = model.f(x_train_normalized[sorted_indices]).detach()

plt.figure()
plt.plot(x_train, y_train, 'o', label='Original data')
plt.plot(x_train_sorted, predictions_sorted, label='Fitted line')
plt.legend()
plt.show()
