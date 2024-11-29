import torch

def test1():
    x = torch.randn(10, 10).to("hpu")
    print("x.is_leaf", x.is_leaf) # x.is_leaf True

    y = torch.randn([10, 10], requires_grad=True).to("hpu")
    print("y.is_leaf", y.is_leaf) # y.is_leaf False

def test2():
    def fn(x, y):
        loss = (x - y).pow(2).sum()
        return loss
    x = torch.randn(10, 10).to("hpu").requires_grad_()
    y = torch.randn(10, 10).to("hpu")
    compiled = torch.compile(fn, backend="hpu_backend")
    loss = compiled(x, y)
    print("END FORWARD", flush=True)
    print("loss", loss)

def test3():
    def fn(x, y):
        loss = (x - y).pow(2).sum()
        return loss
    x = torch.randn(10, 10).to("hpu").requires_grad_()
    y = torch.randn(10, 10).to("hpu")
    compiled = torch.compile(fn, backend="hpu_backend")
    loss = compiled(x, y)
    print("END FORWARD", flush=True)
    loss.backward()
    print("END BACKWARD", flush=True)

    print("x.grad:", x.grad) # there are a lot of eager ops after loss.backward()
    # print("x.grad:", x.grad.to("cpu"))

test3()
