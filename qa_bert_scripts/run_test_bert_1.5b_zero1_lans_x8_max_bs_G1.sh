# changing json file to fit test needs
SCRIPTDIR=$(dirname "$0")
export HL_DS_CONFIG=${QA_DS_CONFIG:-/root/logs/config_tmp.json}
python3 $SCRIPTDIR/overrun_in_json.py -i $SCRIPTDIR/deepspeed_config_bert_1.5b_zero1.json \
       -o ${HL_DS_CONFIG} --train_bs 1024 --micro_bs 32 --steps_per_print 1

export HL_NUM_NODES=${QA_NUM_NODES:-1}
export HL_RUN_STEPS=${QA_RUN_STEPS:-3}
export HL_LOG_FREQ=${QA_LOG_FREQ:-1}
export HL_RESULTS_DIR=${QA_RESULTS_DIR:-"./results/bert_1.5b_zero1_lans_x8_max_bs/"}

$SCRIPTDIR/run_bert_1.5b_32x.sh
