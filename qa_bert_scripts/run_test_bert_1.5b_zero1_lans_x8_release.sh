# Using json and main bash file from external model garden dir
SCRIPTDIR=$(dirname "$0")
export HL_DS_CONFIG=${QA_DS_CONFIG:-/root/logs/config_tmp.json}
export MICRO_BS=${QA_MICRO_BS:-128}
echo "BS = ${MICRO_BS}"
python3 $SCRIPTDIR/overrun_in_json.py -i $SCRIPTDIR/deepspeed_config_bert_1.5b.json \
       -o ${HL_DS_CONFIG} --micro_bs $MICRO_BS

export HL_NUM_NODES=${QA_NUM_NODES:-1}
export HL_RUN_STEPS=${QA_RUN_STEPS:-10}
export HL_LOG_FREQ=${QA_LOG_FREQ:-5}
export HL_RESULTS_DIR=${QA_RESULTS_DIR:-"./results/bert_1.5b_zero1_lans_x8_release/"}

$SCRIPTDIR/run_bert_1.5b_32x.sh