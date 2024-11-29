import torch

class MyTensor(torch.Tensor):
    @classmethod
    def __torch_dispatch__(self, func, *args, **kwargs):
        if (func == torch.ops.aten.add.Tensor):
            print(f"Customized add function called with {len(args)} args")
            return torch.Tensor([1, 2, 3])
        return super().__torch_dispatch__(func, *args, **kwargs)

# Create a custom tensor
custom_tensor = MyTensor([1, 2, 3])

# Call a torch function on the custom tensor
print(custom_tensor.add(1)) # Output: 6