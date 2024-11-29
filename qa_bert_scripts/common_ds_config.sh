#!/bin/bash

if [ "${QA_TORCH_COMPILE}" == "1" ]; then
  tmpfile=$(mktemp XXXXXX.json)
  jq '."compile"."enabled"=true | ."compile"."backend"="hpu_backend"' ${HL_DS_CONFIG} > $tmpfile
  cp $tmpfile ${HL_DS_CONFIG}
  rm $tmpfile
fi
echo deepspeed config file ${HL_DS_CONFIG}
jq --indent 6 . ${HL_DS_CONFIG}
