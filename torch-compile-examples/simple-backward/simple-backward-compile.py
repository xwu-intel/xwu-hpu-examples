import torch

# Define a simple function to compute the loss
def compute_loss(x, y):
    loss = (x - y).pow(2).sum()
    return loss

# Create random tensors
x = torch.randn(10, 10).to("hpu").requires_grad_()
# Why this is wrong for HPU?
# x = torch.randn([10, 10], requires_grad=True).to("hpu")
y = torch.randn(10, 10).to("hpu")

# Compile the loss function
compiled_loss = torch.compile(compute_loss, backend="hpu_backend", fullgraph=True)

print("=== Start Forward pass ===", flush=True)
# Forward pass
loss = compiled_loss(x, y)
print("=== End Forward pass ===", flush=True)

print("=== Start Backward pass ===", flush=True)
# Backward pass
loss.backward()
print("=== End Backward pass ===", flush=True)

# Print the gradients of x
print("x.grad:", x.grad)
