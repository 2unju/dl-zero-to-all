import torch

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

w = torch.zeros(1, requires_grad=True)
lr = 0.1

epochs = 10
for epoch in range(epochs + 1):
    hypothesis = x_train * w

    cost = torch.mean((hypothesis - y_train) ** 2)
    gradient = torch.sum((w * x_train - y_train) * x_train)

    print('Epoch {:4d}/{} W: {:.3f}, Cost: {:.6f}'.format(epoch, epochs, w.item(), cost.item()))

    w.data -= lr * gradient