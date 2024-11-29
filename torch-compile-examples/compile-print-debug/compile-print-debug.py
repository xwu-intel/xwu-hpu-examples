import torch

@torch.library.custom_op("debug::print_op", mutates_args=())
def print_sth(x: torch.Tensor) -> torch.Tensor:
    print(x)
    return x.clone()

@print_sth.register_fake
def _(x) -> torch.Tensor:
    return torch.empty_like(x)

def forward(x: torch.Tensor):
    y = x.sin()
    return print_sth(y)

m = torch.compile(forward)

m(torch.randn((1,3)))
