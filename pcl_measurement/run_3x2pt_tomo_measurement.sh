#!/bin/bash

start=$SECONDS

PIPELINE_VARIABLES_PATH="/raid/scratch/wongj/mywork/3x2pt/SWEPT/pcl_measurement/set_variables_3x2pt_measurement.ini"
export PIPELINE_VARIABLES_PATH

source <(grep = $PIPELINE_VARIABLES_PATH)

echo Creating nz...
python create_nz_boundaries.py
echo Done

for ITER_NO in $(seq 1 $REALISATIONS)
do
  export ITER_NO

  echo Measuring power spectra from mock catalogue - Realisation ${ITER_NO} / ${REALISATIONS}
  python measure_cat_3x2pt_pcls.py

done

echo Calculating Cls averaged over realisations...
python av_cls.py
echo Done

echo Running Cosmosis and calculating theoretical 3x2pt spectra...

export PIPELINE_DIR
export COSMOSIS_ROOT_DIR
export MEASUREMENT_SAVE_DIR
export INPUT_ELL_MIN
export INPUT_ELL_MAX
export MEASURED_NZ_TABLE_NAME

SAVE_DIR=$MEASUREMENT_SAVE_DIR
NZ_TABLE_FILENAME=$MEASURED_NZ_TABLE_NAME
ELL_MAX=$INPUT_ELL_MAX
ELL_MIN=$INPUT_ELL_MIN

export SAVE_DIR
export NZ_TABLE_FILENAME
export ELL_MIN
export ELL_MAX

cd ${PIPELINE_DIR}/software_utils/
bash run_cosmosis.sh  &> ${SAVE_DIR}run_cosmosis_log.txt
echo Done
cd ${PIPELINE_DIR}/pcl_measurement/

GAUSSIAN_CL_LIKELIHOOD_PATH=${PIPELINE_DIR}gaussian_cl_likelihood/
export GAUSSIAN_CL_LIKELIHOOD_PATH

echo Measuring Bandpowers from average Cls...
python measure_cat_bps.py
echo Done

echo Converting 3x2pt Bps to inference routine input...
python conv_bps.py
echo Done

echo Calculating 3x2pt covariance from simulations...
python cov_fromsim.py
echo Done

echo Weak Lensing Tomography Pipeline Complete :\)

duration=$((SECONDS-start))
echo Total time elapsed: $duration

