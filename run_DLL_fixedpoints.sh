#!/bin/bash

model_type="FlowMatching"


for p in 3.0 6.0 9.0
do
  echo "Running DLL for ${p} GeV/c"
  python run_DLL_fixedpoints_hpDIRC.py --config config/hpDIRC_config_clean.json --momentum $p --model_type $model_type
  echo "------------------------------------------------- "
  echo " " 
done