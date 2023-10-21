import torch
import math
import torch.nn.functional as F

# Input data
x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([0.0, 1.0, 1.0, 0.0])

# Initial Weights
w1 = torch.tensor([[0.15, -0.61], [-0.26, 0.35]], requires_grad=True)
b1 = torch.tensor([-0.25, 0.68], requires_grad=True)
w2 = torch.tensor([-0.45, 0.96], requires_grad=True)
b2 = torch.tensor([0.78], requires_grad=True)

epochs = 250 
learning_rate = 0.03
loss_fn = torch.nn.MSELoss()
optim = torch.optim.Adam([w1, b1, w2, b2], lr=learning_rate)

for i in range(epochs):

    # Forward pass
    l1 = x @ w1
    l1b = l1 + b1
    r1 = F.tanh(l1b)
    l2 = r1 @ w2
    l2b = l2 + b2

    loss = loss_fn(l2b, y)

    loss.backward()
    optim.step()
    optim.zero_grad()

    print(f"{i}:{loss:.6f}")

print(f"----------------------------")
print(f"Final loss: {loss:.6f}")
# print(f"Final weights: {w1.detach().numpy()} {b1.detach().numpy()} {w2.detach().numpy()} {b2.detach().numpy()}")
print(f"Final predictions: {[1 if el >= 0.5 else 0 for el in l2b.detach().numpy()]}")
print(f"Expected predictions: {y.numpy()}")