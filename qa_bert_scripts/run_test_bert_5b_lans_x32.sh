# using the same json
SCRIPTDIR=$(dirname "$0")
if [ "$DEVICE_TYPE" = "gaudi2" ]
then
    MICRO_BS=32
else
    MICRO_BS=8
fi
export HL_DS_CONFIG=${QA_DS_CONFIG:-/root/logs/config_tmp.json}
python3 $SCRIPTDIR/overrun_in_json.py -i $SCRIPTDIR/deepspeed_config_bert_5b_lans.json -o ${HL_DS_CONFIG}\
       --micro_bs $MICRO_BS --reduce_bucket_size 100000000

export HL_NUM_NODES=${QA_NUM_NODES:-4}
export HL_RUN_STEPS=${QA_RUN_STEPS:-5}
export HL_LOG_FREQ=${QA_LOG_FREQ:-1}
export HL_RESULTS_DIR=${QA_RESULTS_DIR:-"./results/bert_5b_zero2_lans_x32/"}

$SCRIPTDIR/run_bert_5b_128x_lans.sh
