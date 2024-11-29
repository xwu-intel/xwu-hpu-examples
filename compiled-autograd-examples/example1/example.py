import torch
import os

# env variable to disable lazy mode
os.environ['PT_HPU_LAZY_MODE'] = "0"

import habana_frameworks.torch.hpu
import habana_frameworks.torch.core as htcore

class Model(torch.nn.Module):
   def __init__(self):
      super().__init__()
      self.linear = torch.nn.Linear(10, 10)

   def forward(self, x):
      return self.linear(x)


model = Model().to('hpu')
x = torch.randn(10).to('hpu').requires_grad_()

def my_hook(grad):
   return grad + 2000

torch._dynamo.config.compiled_autograd = True
@torch.compile(backend='hpu_backend')
def train(model, x):
   # print("=============1=================")
   loss = model(x).sum()
   # print("=============2=================", loss)
   # print("=============3=================", my_hook(10))
   loss.backward()

print("start")
train(model, x)
print("end")