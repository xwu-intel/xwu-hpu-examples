import torch
import habana_frameworks.torch.core as htcore
from habana_frameworks.torch.dynamo.compile_backend.experimental import enable_compiled_autograd
import habana_frameworks.torch.hpu as hthpu

import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import os

torch._inductor.config._fuse_ddp_bucket_size = 2048*2048*4/1024/1024 / 4  # 4MB

# world_size = 8
world_size = 2
group = [i for i in range(world_size)]


def print_rank0(str):
    return
    if dist.get_rank() == 0:
        print(str)


def grad_sync_clear(param, keep_grad):
    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
    # future = torch.ops.c10d_functional.all_reduce(
    #     param.grad, "sum", "", [0, 1, 2, 3], 4)
    # future = torch.ops.c10d_functional.all_reduce(
    #     param.grad, "sum", "", group, world_size)
    # waittensor = torch.ops.c10d_functional.wait_tensor(future)
    # future = torch.ops._c10d_functional.all_reduce(
    #     param.grad, "sum", torch.distributed.distributed_c10d._resolve_group_name_by_ranks_and_tag(group, ""))
    # waittensor = torch.ops._c10d_functional.wait_tensor(future)
    # param.grad.copy_(waittensor)
    # waittensor = None
    if not keep_grad:
        param.grad = None


class Module(torch.nn.Module):
    def __init__(self, ioc):
        super().__init__()
        self.fc1 = torch.nn.Linear(ioc, ioc, bias=False).to("hpu")
        self.fc2 = torch.nn.Linear(ioc, ioc, bias=False).to("hpu")
        self.fc3 = torch.nn.Linear(ioc, ioc, bias=False).to("hpu")
        self.fc4 = torch.nn.Linear(ioc, ioc, bias=False).to("hpu")
        self.fc5 = torch.nn.Linear(ioc, ioc, bias=False).to("hpu")
        self.fc6 = torch.nn.Linear(ioc, ioc, bias=False).to("hpu")
        self.fc7 = torch.nn.Linear(ioc, ioc, bias=False).to("hpu")
        self.fc8 = torch.nn.Linear(ioc, ioc, bias=False).to("hpu")

        self.grad_acc_hooks = []
        self.grad_acc = []
        self.params = [self.fc1.weight, self.fc2.weight,
                       self.fc3.weight, self.fc4.weight,
                       self.fc5.weight, self.fc6.weight,
                       self.fc7.weight, self.fc8.weight]
        for i, param in enumerate(self.params):

            keep_grad = False
            if i == dist.get_rank():
                keep_grad = True

            def wrapper(param, keep_grad):
                param_tmp = param.expand_as(param)
                grad_acc = param_tmp.grad_fn.next_functions[0][0]

                def grad_acc_hook(*notneeded):
                    grad_sync_clear(param, keep_grad)

                self.grad_acc.append(grad_acc)
                self.grad_acc_hooks.append(
                    grad_acc.register_hook(grad_acc_hook))

            wrapper(param, keep_grad)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        return x.sum()


def run(args, rank):
    # if rank == 0:
    #     os.environ["LOG_LEVEL_ALL_PT"] = "1"
    #     os.environ["PT_TOWL_LOG_ENABLE"] = "1"

    bs = 128
    ioc = 2048

    model = Module(ioc)

    if args.mode == "compile":
        model_to_train = torch.compile(model, backend="hpu_backend")
    else:
        model_to_train = model

    input = torch.randn([bs, ioc]).to("hpu")

    # run eager as ref
    # loss = model(input)
    # with torch.autograd.grad_mode.set_multithreading_enabled(False):
    #     loss.backward()
    # eager_gradients = []
    # for name, param in model.named_parameters():
    #     if param.requires_grad and param.grad is not None:
    #         eager_gradients.append(param.grad.to("cpu"))
    #         param.grad = None

    # reset max mem
    hthpu.reset_peak_memory_stats()

    # run cag
    if args.mode == "compile" and args.compiled_autograd:
        enable_compiled_autograd()

    loss = model_to_train(input)
    if args.mode == "lazy":
        htcore.mark_step()
    with torch.autograd.grad_mode.set_multithreading_enabled(False):
        loss.backward()
    if args.mode == "lazy":
        htcore.mark_step()
    cag_gradients = []
    for name, param in model_to_train.named_parameters():
        if param.requires_grad and param.grad is not None:
            cag_gradients.append(param.grad)
            param.grad = None

    # check resutls
    # assert len(eager_gradients) == len(cag_gradients)
    # for i in range(len(eager_gradients)):
    #     eager_cpu = eager_gradients[i]
    #     print_rank0("eager_cpu")
    #     print_rank0(eager_cpu)
    #     cag_cpu = cag_gradients[i].to("cpu")
    #     print_rank0("cag_cpu")
    #     print_rank0(cag_cpu)
    #     assert torch.allclose(eager_cpu, cag_cpu, rtol=1e-5, atol=1e-5)

    print(
        f"Max mem allocated on rank {rank} is {hthpu.max_memory_allocated()/1024/1024}")


def init_process(args, size, rank, fn, backend='hccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ["WORLD_SIZE"] = str(size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

    hlog_dir = os.environ['HABANA_LOGS']
    hlog_dir = hlog_dir + '/' + str(rank)
    os.environ['HABANA_LOGS'] = hlog_dir

    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(args, rank)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str,
                        choices=["eager", "compile", "lazy"], default="eager")
    parser.add_argument("--compiled_autograd", action="store_true")
    parsed_args = parser.parse_args()

    if world_size > 1:
        processes = []
        mp.set_start_method("spawn")
        for rank in range(world_size):
            p = mp.Process(target=init_process, args=(
                parsed_args, world_size, rank, run))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        init_process(parsed_args, world_size, 0, run)
