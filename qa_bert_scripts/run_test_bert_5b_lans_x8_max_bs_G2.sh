# using the same json
SCRIPTDIR=$(dirname "$0")
export HL_DS_CONFIG=${QA_DS_CONFIG:-/root/logs/config_tmp.json}
python3 $SCRIPTDIR/overrun_in_json.py -i $SCRIPTDIR/deepspeed_config_bert_5b_lans.json -o ${HL_DS_CONFIG}\
       --micro_bs 32 --train_bs 1024

export HL_NUM_NODES=${QA_NUM_NODES:-1}
export HL_RUN_STEPS=${QA_RUN_STEPS:-5}
export HL_LOG_FREQ=${QA_LOG_FREQ:-1}
export HL_RESULTS_DIR=${QA_RESULTS_DIR:-"./results/bert_5b_zero2_lans_x8_max_bs/"}

$SCRIPTDIR/run_bert_5b_128x_lans.sh
