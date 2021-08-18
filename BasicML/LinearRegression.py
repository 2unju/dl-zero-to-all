import torch
from tqdm import tqdm
from torch import optim

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

w = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([w, b], lr=0.01)

epochs = 1000
for epoch in tqdm(range(1, epochs + 1)):
    hypothesis = x_train * w + b
    cost = torch.mean((hypothesis - y_train) ** 2)

    optimizer.zero_grad()   # gradient 초기화
    cost.backward()         # gradient 계산
    optimizer.step()        # 개선

print(w)
print(b)