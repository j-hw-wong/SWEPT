#!/bin/bash

cd /raid/scratch/wongj/mywork/3x2pt/SWEPT/catalogue_sim/

bash run_cat_sim.sh

cd /raid/scratch/wongj/mywork/3x2pt/SWEPT/pcl_measurement/

bash run_3x2pt_tomo_measurement.sh

cd /raid/scratch/wongj/mywork/3x2pt/SWEPT/inference_analysis/

bash run_inference.sh