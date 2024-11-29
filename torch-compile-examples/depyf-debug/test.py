# test.py
import torch
from torch import _dynamo as torchdynamo
from typing import List

@torch.compile(backend="aot_eager")
def toy_example(a, b):
   x = a / (torch.abs(a) + 1)
   print(a, b)
   if b.sum() < 0:
       b = b * -1
   return x * b

def main():
   breakpoint()
   for _ in range(100):
       toy_example(torch.randn(10), torch.randn(10))

def explain():
   explanation = torch._dynamo.explain(toy_example)(torch.randn(10), torch.randn(10))
   print(explanation)

if __name__ == "__main__":
   # main()
   explain()