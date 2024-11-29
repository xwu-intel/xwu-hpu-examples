# using the same json
SCRIPTDIR=$(dirname "$0")
export HL_DS_CONFIG=${QA_DS_CONFIG:-/root/logs/config_tmp.json}
if [ "$DEVICE_TYPE" = "gaudi2" ]
then
    MICRO_BS=128
else
    MICRO_BS=32
fi
python3 $SCRIPTDIR/overrun_in_json.py -i $SCRIPTDIR/deepspeed_config_bert_1.5b_zero2.json \
       -o ${HL_DS_CONFIG} --micro_bs $MICRO_BS

export HL_NUM_NODES=${QA_NUM_NODES:-2}
export HL_RUN_STEPS=${QA_RUN_STEPS:-100}
export HL_LOG_FREQ=${QA_LOG_FREQ:-10}
export HL_RESULTS_DIR=${QA_RESULTS_DIR:-"./results/bert_1.5b_zero2_lans_x16/"}

$SCRIPTDIR/run_bert_1.5b_32x.sh