import torch
import matplotlib.pyplot as plt

train_x = torch.tensor([[1.0], [0.0]]).reshape(-1, 1)
train_y = torch.tensor([[0.0], [1.0]]).reshape(-1, 1)


class NOTModel:
    def __init__(self):
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def f(self, x):
        return torch.sigmoid(x @ self.W + self.b)

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.f(x), y)


model = NOTModel()

optimizer = torch.optim.SGD([model.W, model.b], lr=0.1)
for epoch in range(10000):
    model.loss(train_x, train_y).backward()
    optimizer.step()
    optimizer.zero_grad()

plt.plot(train_x, train_y, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('x')
plt.ylabel('y')
x = torch.arange(0.0, 1.0, 0.01).reshape(-1, 1)
plt.plot(x, model.f(x).detach(), label='$\\hat y = f(x) = xW+b$')
plt.legend()
plt.show()
