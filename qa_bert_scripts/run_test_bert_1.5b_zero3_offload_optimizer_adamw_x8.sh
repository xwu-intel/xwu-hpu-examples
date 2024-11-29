# using the same json
SCRIPTDIR=$(dirname "$0")
export HL_DS_CONFIG=${QA_DS_CONFIG:-/root/logs/config_tmp.json}
if [ "$DEVICE_TYPE" = "gaudi2" ]
then
    MICRO_BS=128
else
    MICRO_BS=32
fi
python3 $SCRIPTDIR/overrun_in_json.py -i $SCRIPTDIR/deepspeed_config_bert_1.5b_zero3_offload_optimizer_adamw.json \
       -o ${HL_DS_CONFIG} --micro_bs $MICRO_BS --optimizer_type adamw

export HL_NUM_NODES=${QA_NUM_NODES:-1}
export HL_RUN_STEPS=${QA_RUN_STEPS:-6}
export HL_LOG_FREQ=${QA_LOG_FREQ:-3}
export HL_OPTIMIZER=${QA_OPTIMIZER:-"NONE"}
export HL_RESULTS_DIR=${QA_RESULTS_DIR:-"./results/bert_1.5b_adamw_zero3_offload_optimizer_x8/"}

$SCRIPTDIR/run_bert_1.5b_adamw_8x.sh
