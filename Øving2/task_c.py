import torch
import matplotlib.pyplot as plt

x_train = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y_train = torch.tensor([[0.0], [1.0], [1.0], [0.0]])


class XORModel:
    def __init__(self):
        self.W1 = torch.randn(2, 2, requires_grad=True)
        self.b1 = torch.randn(1, 2, requires_grad=True)
        self.W2 = torch.randn(2, 1, requires_grad=True)
        self.b2 = torch.randn(1, 1, requires_grad=True)

    def set_weights_biases(self, W1, b1, W2, b2):
        self.W1.data = W1.data
        self.b1.data = b1.data
        self.W2.data = W2.data
        self.b2.data = b2.data
        # Ensure we still track gradients after assignment
        self.W1.requires_grad = True
        self.b1.requires_grad = True
        self.W2.requires_grad = True
        self.b2.requires_grad = True

    # Predictor
    def f1(self, x):
        return torch.sigmoid(x @ self.W1 + self.b1)

    def f2(self, x):
        return torch.sigmoid(x @ self.W2 + self.b2)

    def f(self, x):
        return self.f2(self.f1(x))

    # Cross Entropy loss
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.f(x), y)


model1 = XORModel()

model2 = XORModel()
model2.set_weights_biases(
    torch.tensor([[10.0, -10.0], [10.0, -10.0]], requires_grad=True),
    torch.tensor([[-5.0, 15.0]], requires_grad=True),
    torch.tensor([[10.0], [10.0]], requires_grad=True),
    torch.tensor([[-15.0]], requires_grad=True)
)

optimizer = torch.optim.SGD([model2.W1, model2.b1, model2.W2, model2.b2], lr=0.1)
for epoch in range(100000):
    model2.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

optimizer = torch.optim.SGD(
    [model1.W1, model1.b1, model1.W2, model1.b2], lr=0.1)
for epoch in range(100000):
    model1.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()


# Visualization
def plot_xor_model(model, title):
    x_values = torch.linspace(0, 1, 100)
    y_values = torch.linspace(0, 1, 100)
    point_x, point_y = torch.meshgrid(x_values, y_values)
    input_grid = torch.stack((point_x.flatten(), point_y.flatten()), dim=1)
    predictions = model.f(input_grid).detach()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_train[:, 0], x_train[:, 1], y_train[:, 0], color='red', label='True Values', s=100)
    ax.plot_surface(point_x.numpy(), point_y.numpy(), predictions.reshape(100, 100).numpy(), color='None', alpha=0.5)

    ax.set_xlabel('X1 Input')
    ax.set_ylabel('X2 Input')
    ax.set_zlabel('Prediction')
    ax.set_title(title)
    ax.legend()
    plt.show()


# print model losses
print("Model 1 loss: %s" % model1.loss(x_train, y_train))
print("Model 2 loss: %s" % model2.loss(x_train, y_train))

plot_xor_model(model1, "...")
plot_xor_model(model2, "...")
