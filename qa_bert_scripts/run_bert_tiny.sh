#!/bin/bash

###############################################################################################
# Example: Pretraining phase 1 of BERT Tiny parameters on single HLS1 box with 1 device.
###############################################################################################

# Params: run_pretraining
SCRIPTDIR=$(dirname "$0")

DATA_DIR=$HL_DATA_DIR_ROOT/data/pytorch/bert/pretraining/hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/books_wiki_en_corpus
MODEL_CONFIG=${HL_MODEL_CONFIG:-"$SCRIPTDIR/bert_tiny_config.json"}
DS_CONFIG=${HL_DS_CONFIG:-"$SCRIPTDIR/deepspeed_config_bert_tiny.json"}
HOSTSFILE=${HL_HOSTSFILE:-"$SCRIPTDIR/hostsfile"}
RESULTS_DIR=${HL_RESULTS_DIR:-"./results/bert_tiny"}
MAX_SEQ_LENGTH=128
MAX_STEPS=100000
RUN_STEPS=${HL_RUN_STEPS:--1}
LR=0.001
WARMUP=0.2843
LOG_FREQ=${HL_LOG_FREQ:-1}
OPTIMIZER=${HL_OPTIMIZER:-"--optimizer=adamw"}
if [ "$OPTIMIZER" == "NONE" ]
then
    OPTIMIZER=""
fi

CHECKPOINTS_ACTIVATION=${HL_CHECKPOINTS_ACTIVATION:-""}

# Params: DeepSpeed
NUM_NODES=1
NGPU_PER_NODE=${HL_NGPU_PER_NODE:-8}

DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

# Check if HL_TORCH_COMPILE_ENABLED and QA_TORCH_COMPILE are both set and non-empty
if [[ -v HL_TORCH_COMPILE_ENABLED && -n "$HL_TORCH_COMPILE_ENABLED" && \
      -v QA_TORCH_COMPILE && -n "$QA_TORCH_COMPILE" ]]; then
        ENABLE_TORCH_COMPILE=--enable_torch_compile
fi


CMD="python3 -u ./run_pretraining.py \
     --steps_this_run $RUN_STEPS \
     $CHECKPOINTS_ACTIVATION \
     --skip_checkpoint \
     --disable_progress_bar \
     --do_train \
     --bert_model=bert-base-uncased \
     --config_file=$MODEL_CONFIG \
     --json-summary=$RESULTS_DIR/dllogger.json \
     --output_dir=$RESULTS_DIR/checkpoints \
     --seed=12439 \
     $OPTIMIZER \
     --input_dir=$DATA_DIR \
     --max_seq_length $MAX_SEQ_LENGTH \
     --max_predictions_per_seq=20 --max_steps=$MAX_STEPS \
     --warmup_proportion=$WARMUP \
     --learning_rate=$LR \
     --log_freq=$LOG_FREQ \
     $ENABLE_TORCH_COMPILE \
     --deepspeed \
     --deepspeed_config=$DS_CONFIG"

#Configure multinode
if [ "$NUM_NODES" -ne "1" -a -f "$HOSTSFILE" ]
then
    MULTINODE_CMD="--hostfile=$HOSTSFILE"
fi

mkdir -p $RESULTS_DIR
deepspeed --num_nodes ${NUM_NODES} \
          --num_gpus ${NGPU_PER_NODE} \
          --no_local_rank \
          --no_python \
          $MULTINODE_CMD \
          /usr/bin/bash -c "$CMD"