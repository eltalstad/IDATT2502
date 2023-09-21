import torch
import matplotlib.pyplot as plt

x_train = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]).reshape(-1, 2)
y_train = torch.tensor([[1.0], [1.0], [1.0], [0.0]])


class NANDModel:
    def __init__(self):
        self.W = torch.rand((2, 1), requires_grad=True)
        self.b = torch.rand((1, 1), requires_grad=True)

    def f(self, x):
        return torch.sigmoid(x @ self.W + self.b)

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.f(x), y)


model = NANDModel()

optimizer = torch.optim.SGD([model.W, model.b], lr=0.1)
for epoch in range(10000):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

x_values = torch.linspace(0, 1, 100)
y_values = torch.linspace(0, 1, 100)
point_x, point_y = torch.meshgrid(x_values, y_values)

input_grid = torch.stack((point_x.flatten(), point_y.flatten()), dim=1)
predictions = model.f(input_grid).detach()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x_train[:, 0], x_train[:, 1], y_train[:, 0], color='red', label='True Values', s=100)

ax.plot_surface(point_x.numpy(), point_y.numpy(), predictions.reshape(100, 100), color='None', alpha=0.5)

ax.set_xlabel('X1 Input')
ax.set_ylabel('X2 Input')
ax.set_zlabel('Prediction')
ax.set_title('NAND Function')
ax.legend()
plt.show()
