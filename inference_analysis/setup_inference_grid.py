"""
Set up grid of (w0-wa) values to perform likelihood routine.
"""

import os
import configparser
import sys

angular_binning_path = os.environ['ANGULAR_BINNING_PATH']
gaussian_cl_likelihood_path = os.environ['GAUSSIAN_CL_LIKELIHOOD_PATH']
pipeline_dir = os.environ['PIPELINE_DIR']

sys.path.insert(1, angular_binning_path)
sys.path.insert(1, gaussian_cl_likelihood_path)
sys.path.insert(1, pipeline_dir)

from gaussian_cl_likelihood.python import cosmosis_utils, simulation

pipeline_variables_path = os.environ['PIPELINE_VARIABLES_PATH']

config = configparser.ConfigParser()
config.read(pipeline_variables_path)

save_dir = str(config['inference_analysis_params']['MEASUREMENT_SAVE_DIR'])
n_chains = int(float(config['inference_analysis_params']['N_CHAINS']))

params = {
    'cosmological_parameters--w': {
        'min': -1.3,
        'max': -0.7,
        'steps': 4
    },
    'cosmological_parameters--wa': {
        'min': -0.75,
        'max': 0.75,
        'steps': 4
    }
}

output_dir = save_dir + 'inference_analysis/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cosmosis_utils.generate_chain_input(params, n_chains, output_dir)
