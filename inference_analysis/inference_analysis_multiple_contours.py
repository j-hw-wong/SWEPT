"""
Script to plot multiple posterior distributions on the same plot. UNDER CONSTRUCTION, UNSTABLE.
"""

import configparser
import numpy as np
import os
import sys

angular_binning_path = os.environ['ANGULAR_BINNING_PATH']
gaussian_cl_likelihood_path = os.environ['GAUSSIAN_CL_LIKELIHOOD_PATH']
pipeline_dir = os.environ['PIPELINE_DIR']

sys.path.insert(1, angular_binning_path)
sys.path.insert(1, gaussian_cl_likelihood_path)
sys.path.insert(1, pipeline_dir)

from gaussian_cl_likelihood.python import cosmosis_utils, simulation, posteriors
from angular_binning import loop_likelihood_nbin, posterior, mask, param_grids, covariance, error_vs_nbin, \
    like_bp_gauss_mix


pipeline_variables_path = os.environ['PIPELINE_VARIABLES_PATH']

config = configparser.ConfigParser()
config.read(pipeline_variables_path)

output_lmin = int(float(config['inference_analysis_params']['OUTPUT_ELL_MIN']))
output_lmax = int(float(config['inference_analysis_params']['OUTPUT_ELL_MAX']))

n_bandpowers = int(config['inference_analysis_params']['N_BANDPOWERS'])

master_folder = '/raid/scratch/wongj/mywork/3x2pt/FINAL_DATA/NOISE/EQUIPOP/3x2pt/'

contour_1BIN = posterior.get_contours(
    log_like_filemask=master_folder + '1BIN.txt',
    contour_levels_sig=[1, 2],
    bp=n_bandpowers,
    colour='C0',
    linestyle='-'
)

contour_2BIN = posterior.get_contours(
    log_like_filemask=master_folder + '2BIN.txt',
    contour_levels_sig=[1, 2],
    bp=n_bandpowers,
    colour='C1',
    linestyle='-'
)

contour_3BIN = posterior.get_contours(
    log_like_filemask=master_folder + '3BIN.txt',
    contour_levels_sig=[1, 2],
    bp=n_bandpowers,
    colour='C2',
    linestyle='-'
)

contour_4BIN = posterior.get_contours(
    log_like_filemask=master_folder + '4BIN.txt',
    contour_levels_sig=[1, 2],
    bp=n_bandpowers,
    colour='C3',
    linestyle='-'
)

contour_5BIN = posterior.get_contours(
    log_like_filemask=master_folder + '5BIN.txt',
    contour_levels_sig=[1, 2],
    bp=n_bandpowers,
    colour='C4',
    linestyle='-'
)

contour_6BIN = posterior.get_contours(
    log_like_filemask=master_folder + '6BIN.txt',
    contour_levels_sig=[1, 2],
    bp=n_bandpowers,
    colour='C5',
    linestyle='-'
)

contour_7BIN = posterior.get_contours(
    log_like_filemask=master_folder + '7BIN.txt',
    contour_levels_sig=[1, 2],
    bp=n_bandpowers,
    colour='C6',
    linestyle='-'
)

contour_8BIN = posterior.get_contours(
    log_like_filemask=master_folder + '8BIN.txt',
    contour_levels_sig=[1, 2],
    bp=n_bandpowers,
    colour='C7',
    linestyle='-'
)

contour_9BIN = posterior.get_contours(
    log_like_filemask=master_folder + '9BIN.txt',
    contour_levels_sig=[1, 2],
    bp=n_bandpowers,
    colour='C8',
    linestyle='-'
)

contour_10BIN = posterior.get_contours(
    log_like_filemask=master_folder + '10BIN.txt',
    contour_levels_sig=[1, 2],
    bp=n_bandpowers,
    colour='red',
    linestyle='-'
)

contour_10BIN_1 = posterior.get_contours(
    log_like_filemask=master_folder + '10BIN.txt',
    contour_levels_sig=[1],
    bp=n_bandpowers,
    colour='red',
    linestyle='-'
)

contour_10BIN_2 = posterior.get_contours(
    log_like_filemask=master_folder + '10BIN.txt',
    contour_levels_sig=[2],
    bp=n_bandpowers,
    colour='red',
    linestyle='-'
)

'''
posterior.plot_multiple_contours(
    contour_params=[contour_1BIN, contour_10BIN_1, contour_10BIN_2],
    title='',
    labels=['No Photo-z', 'With Photo-z'],
    plot_save_path=master_folder+'contours_photoz_corr.png',
)
'''

posterior.plot_multiple_contours(
    contour_params=[contour_1BIN, contour_2BIN, contour_3BIN],
    #title='w0-wa 3x2pt Constraints\nfor Equipopulated Bins (No Noise)',
    title=' ',
    labels=['1 Bin', '2 Bins', '3 Bins'],
    plot_save_path=master_folder+'contours_equipop_noise_123_3x2pt.png',
)


'''
posterior.plot_multiple_contours(
    contour_params=[contour_1BIN, contour_2BIN, contour_3BIN, contour_4BIN, contour_5BIN],
    title='w0-wa 1x2pt E Constraints\nfor Equipopulated Bins',
    labels=['1 Bin', '2 Bins', '3 Bins', '4 Bins', '5 Bins'],
    plot_save_path=master_folder+'contours_l{}-{}_1-5_diag_new.png'.format(output_lmin, output_lmax),
)

posterior.plot_multiple_contours(
    contour_params=[contour_2BIN, contour_4BIN, contour_6BIN, contour_8BIN, contour_10BIN],
    title='w0-wa 1x2pt E Constraints\nfor Equipopulated Bins',
    labels=['2 Bins', '4 Bins', '6 Bins', '8 Bins', '10 Bins'],
    plot_save_path=master_folder+'contours_l{}-{}_evens_diag_new.png'.format(output_lmin, output_lmax),
)
'''