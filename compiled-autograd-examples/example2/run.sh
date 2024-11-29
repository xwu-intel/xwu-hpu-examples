export PT_HPU_LAZY_MODE=0
export TORCH_LOGS_FORMAT="%(levelname)s: %(message)s"
# export TORCH_LOGS="+dynamo,compiled_autograd,aot_graphs,graph_breaks,recompiles"
export TORCH_LOGS="+dynamo,compiled_autograd_verbose"
python test_grad_clear.py --mode compile --compiled_autograd