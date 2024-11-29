import torch
import os

os.environ['PT_HPU_LAZY_MODE'] = "0"
import habana_frameworks.torch.core as htcore

@torch.compile(backend="hpu_backend")
def toy_example(a, b):
   x = a / (torch.abs(a) + 1)
   if b.sum() < 0:
       b = b * -1
   return x * b

def main():
   for _ in range(100):
       toy_example(torch.randn(10).to("hpu"), torch.randn(10).to("hpu"))

if __name__ == "__main__":
   main()
