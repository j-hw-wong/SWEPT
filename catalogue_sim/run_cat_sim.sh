#!/bin/bash

start=$SECONDS

PIPELINE_VARIABLES_PATH="/raid/scratch/wongj/mywork/3x2pt/SWEPT/catalogue_sim/set_variables_cat.ini"
export PIPELINE_VARIABLES_PATH

echo Initialising random galaxy sample...
python init_rand_sample.py
echo Done

echo Creating nz distribution...
python create_nz.py
echo Done

echo Running Cosmosis and calculating theoretical 3x2pt spectra...

source <(grep = $PIPELINE_VARIABLES_PATH)
export PIPELINE_DIR
export COSMOSIS_ROOT_DIR
export SIMULATION_SAVE_DIR
export INPUT_ELL_MIN
export INPUT_ELL_MAX
export NZ_TABLE_NAME

SAVE_DIR=$SIMULATION_SAVE_DIR
NZ_TABLE_FILENAME=$NZ_TABLE_NAME
ELL_MAX=$INPUT_ELL_MAX
ELL_MIN=$INPUT_ELL_MIN

export SAVE_DIR
export NZ_TABLE_FILENAME
export ELL_MIN
export ELL_MAX

cd ${PIPELINE_DIR}/software_utils/
bash run_cosmosis.sh  &> ${SAVE_DIR}run_cosmosis_log.txt
echo Done
cd ${PIPELINE_DIR}/catalogue_sim/

echo Converting Cosmosis output to Flask input format...
python conv_fields_cosmosis_flask.py
echo Done

export REALISATIONS

for ITER_NO in $(seq 1 $REALISATIONS)
do
  export ITER_NO

  mkdir -p ${SAVE_DIR}flask/output/iter_${ITER_NO}/
  echo Running Flask to simulate real-sky field maps - Realisation ${ITER_NO} / ${REALISATIONS}
  python run_flask.py &> ${SAVE_DIR}flask/output/iter_${ITER_NO}/flask_out.txt
  echo Flask simulation complete

  echo Interpolating Clustering, Convergence and Shear Maps - Realisation ${ITER_NO} / ${REALISATIONS}
  python interp_maps.py
  echo Done

  echo Randomly sampling galaxy positions from map - Realisation ${ITER_NO} / ${REALISATIONS}
  python poisson_sample_gal_position.py
  echo Done

  echo Compiling mock galaxy catalogue - Realisation ${ITER_NO} / ${REALISATIONS}
  python compile_cat.py
  echo Done

  echo CLEANING
  python clean_products.py - Realisation ${ITER_NO} / ${REALISATIONS}
  echo Done

done

echo Catalogue Simulation Pipeline Complete :\)

duration=$((SECONDS-start))
echo Total time elapsed: $duration


