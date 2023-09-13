import torch
import torch.nn as nn
import torchvision
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

mnist_train = torchvision.datasets.FashionMNIST('./data/fashion', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 1, 28, 28).float()  # torch.functional.nn.conv2d argument must include channels (1)
y_train = torch.zeros((mnist_train.targets.shape[0], 10))  # Create output tensor
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1  # Populate output

mnist_test = torchvision.datasets.FashionMNIST('./data/fashion', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 1, 28, 28).float()  # torch.functional.nn.conv2d argument must include channels (1)
y_test = torch.zeros((mnist_test.targets.shape[0], 10))  # Create output tensor
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1  # Populate output

# Normalization of inputs
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Divide training data into batches to speed up optimization
batches = 600
x_train_batches = torch.split(x_train, batches)
y_train_batches = torch.split(y_train, batches)

# Define model
class ConvolutionalNeuralNetworkModel(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetworkModel, self).__init__()

        self.logits = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),

            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 10))

    # Predictor
    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy loss
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


model = ConvolutionalNeuralNetworkModel().to(device)

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.Adam(model.parameters(), 0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

for epoch in range(20):
    for batch in range(len(x_train_batches)):
        model.loss(x_train_batches[batch].to(device), y_train_batches[batch].to(device)).backward()  # Compute loss gradients
        optimizer.step()  # Perform optimization by adjusting W and b,
        optimizer.zero_grad()  # Clear gradients for next step

    scheduler.step()

    print("accuracy = %s" % model.accuracy(x_test.to(device), y_test.to(device)))

