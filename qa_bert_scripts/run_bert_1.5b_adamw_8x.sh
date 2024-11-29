#!/bin/bash

##########################################################################################
# Example: Pretraining phase 1 of BERT with 1.5B parameters on multinode with 8 card each
##########################################################################################

SCRIPTDIR=$(dirname "$0")

# Params: run_pretraining
DATA_DIR=$HL_DATA_DIR_ROOT/data/pytorch/bert/pretraining/hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/books_wiki_en_corpus
MODEL_CONFIG=${HL_MODEL_CONFIG:-"$SCRIPTDIR/bert_1.5b_config.json"}
DS_CONFIG=${HL_DS_CONFIG:-"$SCRIPTDIR/deepspeed_config_bert_1.5b.json"}
HOSTSFILE=${HL_HOSTSFILE:-"$SCRIPTDIR/hostsfile"}
RESULTS_DIR=${HL_RESULTS_DIR:-"./results/bert_1.5b_adamw"}
MAX_SEQ_LENGTH=128
MAX_STEPS=2000000
RUN_STEPS=${HL_RUN_STEPS:--1}
LR=1e-4
LOG_FREQ=${HL_LOG_FREQ:-10}
MAX_PRED=20
# Params: DeepSpeed
NUM_NODES=${HL_NUM_NODES:-1}
NGPU_PER_NODE=${HL_NGPU_PER_NODE:-8}
OPTIMIZER=${HL_OPTIMIZER:-"--optimizer=adamw"}
if [ "$OPTIMIZER" == "NONE" ]
then
    OPTIMIZER=""
fi

DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)


# Check if HL_TORCH_COMPILE_ENABLED and QA_TORCH_COMPILE are both set and non-empty
if [[ -v HL_TORCH_COMPILE_ENABLED && -n "$HL_TORCH_COMPILE_ENABLED" && \
      -v QA_TORCH_COMPILE && -n "$QA_TORCH_COMPILE" ]]; then
        ENABLE_TORCH_COMPILE=--enable_torch_compile
fi


CMD="python3 -u ./run_pretraining.py \
     $OPTIMIZER \
     --scheduler_degree=1.0 \
     --skip_checkpoint \
     --do_train \
     --bert_model=bert-base-uncased \
     --config_file=$MODEL_CONFIG \
     --json-summary=$RESULTS_DIR/dllogger.json \
     --output_dir=$RESULTS_DIR/checkpoints \
     --seed=12439 \
     --input_dir=$DATA_DIR \
     --max_seq_length $MAX_SEQ_LENGTH \
     --max_predictions_per_seq=$MAX_PRED \
     --max_steps=$MAX_STEPS \
     --steps_this_run=$RUN_STEPS \
     --learning_rate=$LR \
     --disable_progress_bar \
     --log_freq=$LOG_FREQ \
     $ENABLE_TORCH_COMPILE \
     --deepspeed \
     --deepspeed_config=$DS_CONFIG"

#Configure multinode
if [ "$NUM_NODES" -ne "1" -a -f "$HOSTSFILE" ]
then
    MULTINODE_CMD="--hostfile=$HOSTSFILE \
                   --master_addr $(head -n 1 $HOSTSFILE | sed -n s/[[:space:]]slots.*//p) "
fi

mkdir -p $RESULTS_DIR
deepspeed --num_nodes ${NUM_NODES} \
          --num_gpus ${NGPU_PER_NODE} \
          --no_local_rank \
          --no_python \
          $MULTINODE_CMD \
          $CMD
