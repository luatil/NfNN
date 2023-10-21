import torch

# Input data
x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([0.0, 1.0, 1.0, 0.0])

w1 = torch.tensor([[0.15, -0.61], [-0.26, 0.35]], requires_grad=True)
b1 = torch.tensor([-0.25, 0.68], requires_grad=True)
w2 = torch.tensor([-0.45, 0.96], requires_grad=True)
b2 = torch.tensor([0.78], requires_grad=True)

epochs = 32
learning_rate = 0.01

for i in range(epochs):
    l1 = torch.matmul(x, w1)
    l1b = l1 + b1
    r1 = torch.nn.functional.relu(l1b)
    l2 = torch.matmul(r1, w2)
    l2b = l2 + b2
    diff = l2b - y 
    square = diff ** 2
    loss = torch.sum(square)

    print(f"{i}:{loss:.6f}")

    loss.backward()

    w1.data = w1 - learning_rate * w1.grad
    w2.data = w2 - learning_rate * w2.grad
    b1.data = b1 - learning_rate * b1.grad
    b2.data = b2 - learning_rate * b2.grad

    w1.grad.zero_()
    w2.grad.zero_()
    b1.grad.zero_()
    b2.grad.zero_()