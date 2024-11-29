export TORCH_LOGS_FORMAT="%(levelname)s: %(message)s"
# export TORCH_LOGS="+dynamo,compiled_autograd,aot_graphs,graph_breaks,recompiles"
export TORCH_LOGS="+dynamo,compiled_autograd_verbose,graph_breaks,recompiles"
export TORCH_COMPILE_DEBUG=1
python example.py