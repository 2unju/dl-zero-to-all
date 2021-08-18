import numpy as np
import torch

t = np.array([0., 1., 2., 3., 4., 5., 6.])
print(t)
print()

print('Rank of t: ', t.ndim)
print('Shape of t: ', t.shape)
print()

print('t[0] t[1] t[-1] = ', t[0], t[1], t[-1])
print('t[2:5] t[4:-1]  = ', t[2:5], t[4:-1])
print('t[:2] t[3:]     = ', t[:2], t[3:])
print()

t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
print(t)
print()

print('Rank of t: ', t.ndim)
print('Shape of t: ', t.shape)
print()

print('t[0] t[1] t[-1] = ', t[0], t[1], t[-1])
print('t[2:5] t[4:-1]  = ', t[2:5], t[4:-1])
print('t[:2] t[3:]     = ', t[:2], t[3:])
print()

t = torch.FloatTensor([[1., 2., 3.],
                       [4., 5., 6.],
                       [7., 8., 9.],
                       [10., 11., 12.]])
print(t)
print()

print(t.dim())
print(t.size())
print(t[:, 1])
print(t[:, 1].size())
print(t[:, :-1])
print()


# Broadcasting
# Same shape
m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([[2, 2]])
print('{} + {} = {}'.format(m1, m2, m1 + m2))

# Vector + scalar
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([[3]])       # 3 -> [[3, 3]]
print('{} + {} = {}'.format(m1, m2, m1 + m2))

# 2 x 1 vector + 1 x 2 vector
m1 = torch.FloatTensor([[1, 2]])    # [[1, 2]] -> [[1, 2], [1, 2]]
m2 = torch.FloatTensor([[3], [4]])  # [[3], [4]] -> [[3, 3], [4, 5]]
print('{} + {} = {}'.format(m1, m2, m1 + m2))
print()

# Mean
lt = torch.LongTensor([1, 2])
try:
    print(t.mean())
except Exception as exc:
    print(exc)
print()

print(t)
print()
print(t.mean())
print(t.mean(dim=0))
print(t.mean(dim=1))
print(t.mean(dim=-1))
print()

print(t.sum())
print(t.sum(dim=0))
print(t.sum(dim=1))
print()

# Max and Argmax
print(t.max())
print(t.max(dim=1))
print('Max: ', t.max(dim=0)[0])
print('Argmax: ', t.max(dim=0)[1])