import torch
import pandas as pd
import matplotlib.pyplot as plt

# Read data into a pandas dataframe
data = pd.read_csv('day_length_weight.csv')
x_train = torch.tensor(data[['length', 'weight']].values, dtype=torch.float32)  # Notice the double brackets for selecting multiple columns
y_train = torch.tensor(data['day'].values, dtype=torch.float32).reshape(-1, 1)

class LinearRegressionModel:

    def __init__(self):
        # Model variables
        self.W = torch.randn([2, 1], requires_grad=True)
        self.b = torch.randn([1, 1], requires_grad=True)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b  # @ corresponds to matrix multiplication

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)  # Can also use torch.nn.functional.mse_loss(self.f(x), y) to possibly increase numberical stability


model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], 0.0001)
for epoch in range(1000000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result in 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['length'], data['weight'], y_train, label='Data', color='blue')
ax.set_xlabel('Length')
ax.set_ylabel('Weight')
ax.set_zlabel('Day')

# For plotting the plane
x0 = torch.linspace(torch.min(x_train[:, 0]), torch.max(x_train[:, 0]), 100)
x1 = torch.linspace(torch.min(x_train[:, 1]), torch.max(x_train[:, 1]), 100)
X0, X1 = torch.meshgrid(x0, x1)
X = torch.cat([X0.reshape(-1, 1), X1.reshape(-1, 1)], dim=1)
Y = model.f(X).detach()

ax.plot_surface(X0.numpy(), X1.numpy(), Y.reshape(100, 100).numpy(), color='green', alpha=0.5, label='Predicted Plane')
plt.show()

