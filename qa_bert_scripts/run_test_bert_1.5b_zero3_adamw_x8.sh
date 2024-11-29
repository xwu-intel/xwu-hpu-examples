# using the same json
SCRIPTDIR=$(dirname "$0")
export HL_DS_CONFIG=${QA_DS_CONFIG:-/root/logs/config_tmp.json}
export HL_OVERLAP_COMM=${QA_OVERLAP_COMM:-false}
#export HL_USE_ALL_REDUCE=${QA_USE_ALL_REDUCE:-false}
export MICRO_BS=${QA_MICRO_BS:-128}
echo "BS = ${MICRO_BS}"

if [ ! -z "${QA_USE_ALL_REDUCE}" ]; then
  python3 $SCRIPTDIR/overrun_in_json.py -i $SCRIPTDIR/deepspeed_config_bert_1.5b_zero3_adamw.json \
       -o ${HL_DS_CONFIG} --micro_bs $MICRO_BS --overlap_comm ${HL_OVERLAP_COMM} --use_all_reduce ${QA_USE_ALL_REDUCE}
else
  python3 $SCRIPTDIR/overrun_in_json.py -i $SCRIPTDIR/deepspeed_config_bert_1.5b_zero3_adamw.json \
       -o ${HL_DS_CONFIG} --micro_bs $MICRO_BS --overlap_comm ${HL_OVERLAP_COMM}
fi

export HL_NUM_NODES=${QA_NUM_NODES:-1}
export HL_RUN_STEPS=${QA_RUN_STEPS:-10}
export HL_LOG_FREQ=${QA_LOG_FREQ:-5}
export HL_RESULTS_DIR=${QA_RESULTS_DIR:-"./results/bert_1.5b_adamw_zero3_x8/"}

$SCRIPTDIR/run_bert_1.5b_adamw_8x.sh
