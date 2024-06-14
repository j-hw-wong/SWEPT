#!/bin/bash

# Bash script to submit N_CHAINS cosmosis chains using nohup,
# with output from each chain to a separate log file.
# Pipeline must be set up to use the CHAIN_NO environment variable.

# Parameters to set before using
N_CHAINS=${N_CHAINS}

COSMOSIS_DIR=$COSMOSIS_ROOT_DIR
PIPELINE_PATH=${GAUSSIAN_CL_LIKELIHOOD_PATH}ini/tomo_3x2_pipeline_obs.ini
export NZ_TABLE_PATH

cd $COSMOSIS_DIR

source config/setup-cosmosis

# Loop
for ((CHAIN_NO = 0; CHAIN_NO < $N_CHAINS; CHAIN_NO++))
do
   CHAIN_LOG="chain$CHAIN_NO.out"
   export CHAIN_NO
   nohup cosmosis $PIPELINE_PATH > $CHAIN_LOG 2>&1 &
   echo "Chain $CHAIN_NO running with PID $!"
done

# Tidy up
unset CHAIN_NO
echo "Done"

wait
