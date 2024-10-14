import torch

# 创建一个形状为 (2, 3) 的张量
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("原始张量:")
print(x)

# 使用 view() 将其重塑为 (3, 2)
y = x.view(3, 2)
print("重塑后的张量:")
print(y)
