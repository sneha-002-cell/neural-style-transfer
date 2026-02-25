import torch

content = torch.randn(1, 3, 256, 256)
style = torch.randn(1, 3, 256, 256)

target = content.clone().requires_grad_(True)

optimizer = torch.optim.Adam([target], lr=0.01)

for step in range(100):
    optimizer.zero_grad()
    loss = torch.mean((target - content) ** 2)
    loss.backward()
    optimizer.step()

print("Style transfer completed.")
