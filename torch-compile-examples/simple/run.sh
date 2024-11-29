# export LOG_FILE_SIZE=1048576000
export HABANA_LOGS=./.habana_logs

# Enable logs
# export LOG_LEVEL_ALL=1
# export LOG_LEVEL_ALL_PT=1
# export ENABLE_CONSOLE=1

# Dump bridge graph
# pip install pydot protobuf
export PT_HPU_GRAPH_DUMP_MODE=compile_fx

# Dump synapse graph
# export GRAPH_VISUALIZATION=1

export PT_HPU_LAZY_MODE=0

# Disable specific pass
# export PT_HPU_DISABLE_pass_handle_view_before_inplace_compute_ops=True

rm -rf *.pbtxt .graph_dumps .habana_logs torch_compile_debug

export TORCH_LOGS_FORMAT="%(levelname)s: %(message)s"

# Need to install graphvis
# export TORCH_LOGS="graph

# export TORCH_LOGS="+dynamo,graph,graph_code,graph_breaks,recompiles,aot_graphs,aot_joint_graph,compiled_autograd"
# export TORCH_LOGS="+dynamo,graph_breaks,recompiles"
# export TORCH_LOGS="graph,graph_code,compiled_autograd"
# export TORCH_LOGS="graph"
# export TORCHDYNAMO_VERBOSE=1

# Dump to torch_compile_debug directory
# export TORCH_COMPILE_DEBUG=1

python simple.py
