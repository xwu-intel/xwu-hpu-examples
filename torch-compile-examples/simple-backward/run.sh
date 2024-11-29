export LOG_FILE_SIZE=1048576000
export HABANA_LOGS=./.habana_logs

# Enable logs
export LOG_LEVEL_ALL_PT=1
export ENABLE_CONSOLE=1

# Dump bridge graph
# pip install pydot protobuf
export PT_HPU_GRAPH_DUMP_MODE=all

# Dump synapse graph
export GRAPH_VISUALIZATION=1

export PT_HPU_LAZY_MODE=0

rm -rf *.pbtxt .graph_dumps .habana_logs torch_compile_debug

export PT_HPU_EAGER_PIPELINE_ENABLE=False

export TORCH_LOGS="+dynamo,graph,graph_code,graph_breaks,recompiles,aot_graphs,aot_joint_graph,compiled_autograd"
export TORCH_COMPILE_DEBUG=1

python simple-backward-compile.py |& tee simple-backward-compile.log
# python simple-forward-compile.py |& tee simple-forward-compile.log
# python simple-backward-eager.py |& tee simple-backward-eager.log

