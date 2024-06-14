#!/bin/bash

start=$SECONDS

PIPELINE_VARIABLES_PATH="/raid/scratch/wongj/mywork/3x2pt/SWEPT/inference_analysis/set_variables_inference.ini"
export PIPELINE_VARIABLES_PATH

source <(grep = $PIPELINE_VARIABLES_PATH)
export MEASUREMENT_SAVE_DIR
export NZ_TABLE_FILENAME
export COSMOSIS_ROOT_DIR
export INPUT_ELL_MIN
export INPUT_ELL_MAX
export N_CHAINS

GAUSSIAN_CL_LIKELIHOOD_PATH=${PIPELINE_DIR}gaussian_cl_likelihood/
ANGULAR_BINNING_PATH=${PIPELINE_DIR}angular_binning/
export GAUSSIAN_CL_LIKELIHOOD_PATH
export ANGULAR_BINNING_PATH
export PIPELINE_DIR

echo Setting up parameter grid...
python setup_inference_grid.py
echo Done

echo Running CosmosSIS parameter chains...
bash run_inference_chains.sh
echo Done

echo Performing Inference Analysis...
python inference_analysis.py
echo Done

duration=$((SECONDS-start))
echo Total time elapsed: $duration
