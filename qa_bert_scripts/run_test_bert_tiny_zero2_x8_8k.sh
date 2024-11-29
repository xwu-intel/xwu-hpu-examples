# using the same json
SCRIPTDIR=$(dirname "$0")
PARAMS=${QA_NON_DEFAULT_DS_PARAMETER:-}
export HL_DS_CONFIG=${QA_DS_CONFIG:-/root/logs/config_tmp.json}
python3 $SCRIPTDIR/overrun_in_json.py -i $SCRIPTDIR/deepspeed_config_bert_tiny_zero2.json -o ${HL_DS_CONFIG} ${PARAMS}

export HL_NUM_NODES=${QA_NUM_NODES:-1}
export HL_RUN_STEPS=${QA_RUN_STEPS:-8000}
export HL_LOG_FREQ=${QA_LOG_FREQ:-100}
export HL_RESULTS_DIR=${QA_RESULTS_DIR:-"./results/bert_tiny_zero2_x8/"}

$SCRIPTDIR/run_bert_tiny.sh