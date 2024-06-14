"""
Run the Flask software using the pyFlask wrapper to generate the correlated 3x2pt field maps from the 3x2pt power
spectra. Repeated over a given number of realisations/iterations.
"""

import os
import random
import pyFlask
import configparser
import numpy as np


def flask_config(pipeline_variables_path):

    """
    Set up a config dictionary to run based on catalogue simulation pipeline parameters specified in a given
    input variables file

    Parameters
    ----------
    pipeline_variables_path (str):  Path to location of set_variables_cat.ini file

    Returns
    -------
    Dictionary of pipeline parameters to be input into Flask
    """

    config = configparser.ConfigParser()
    config.read(pipeline_variables_path)

    pipeline_dir = str(config['run_pipeline']['PIPELINE_DIR'])
    save_dir = str(config['simulation_setup']['SIMULATION_SAVE_DIR'])
    nbins = int(float(config['create_nz']['N_ZBIN']))

    nside = int(float(config['simulation_setup']['NSIDE']))
    ell_min = float(config['simulation_setup']['INPUT_ELL_MIN'])
    ell_max = float(config['simulation_setup']['INPUT_ELL_MAX'])
    iter_no = int(float(os.environ['ITER_NO']))

    # Prepare config dictionary
    config_dict = {
        'pipeline_dir': pipeline_dir,
        'save_dir': save_dir,
        'nbins': nbins,
        'nside': nside,
        'ell_min': ell_min,
        'ell_max': ell_max,
        'iter_no': iter_no
    }

    return config_dict


def main():

    """
    Run pyFlask for a given random seed and simulation parameters. A dummy flask.config file is read in from the
    pipeline directory and overwritten with the parameters from the dictionary set up in flask_config
    """

    rnd_seed = random.randint(100,999)

    pipeline_variables_path = os.environ['PIPELINE_VARIABLES_PATH']
    config_dict = flask_config(pipeline_variables_path=pipeline_variables_path)

    pipeline_dir = config_dict['pipeline_dir']
    save_dir = config_dict['save_dir']
    nbins = config_dict['nbins']
    nside = config_dict['nside']
    ell_min = config_dict['ell_min']
    ell_max = config_dict['ell_max']
    iter_no = config_dict['iter_no']

    flask_output_dir = save_dir + 'flask/output/' + 'iter_{}/'.format(iter_no)
    if not os.path.exists(flask_output_dir):
        os.makedirs(flask_output_dir)

    np.savetxt(flask_output_dir+'rnd_seed.txt', np.array([rnd_seed]))

    pyFlask.flask([
        "flask", pipeline_dir + "software_utils/flask_3x2pt.config",
        "DIST:", "GAUSSIAN",
        "RNDSEED:", str(rnd_seed),
        "POISSON:", "1",
        "OMEGA_m:", "0.3",
        "OMEGA_L:", "0.7",
        "W_de:", "-1.0",
        "ELLIP_SIGMA:", "0",
        "GALDENSITY:", "0",
        "FIELDS_INFO:", save_dir + "flask/data/field_info_3x2pt.dat",
        "CHOL_IN_PREFIX:", "0",
        "CL_PREFIX:", save_dir + "flask/data/Cl-",
        "ALLOW_MISS_CL:", "1",  # 0 - Return error if required Cl is missing; 1 - Set missing Cls to zero.
        "SCALE_CLS:", "1.0",  # Constant re-scaling factor for all Cls.
        "WINFUNC_SIGMA:", "-1",  # Std. Dev. (arcmin) of field Gaussian smoothing, applied to C(l)s, set <0 for none.
        "APPLY_PIXWIN:", "0",  # Apply (1) or not (0) Healpix pixel window function to C(l)s.
        "SUPPRESS_L:", "-1000",  # l scale for generic exponential suppression of C(l), set <0 for none.
        "SUP_INDEX:", "-6",  # Index for the exponential suppression, exp(-(l/SUPPRESS_L)^SUP_INDEX), <0 for none.
        "REDUCED_SHEAR:", "1",  # Whether to use shear (0) or reduced shear (1) for calculating the observed ellipticities.
        "SELEC_SEPARABLE:", "0",  # Non-sep. sel. func. (0), or separable with common (1) or distinct (2) angular part?
        "SELEC_PREFIX:", "0",  # Fields selection functions FITS files prefix; one file if separable; 0 for full sky.
        "SELEC_Z_PREFIX:", "0",  # Prefix for radial selection functions f(z), one for each galaxy field.
        "SELEC_SCALE:", "0",  # Overall factor to be applied to selection function (1 for no effect).
        "SELEC_TYPE:", "0",  # 0 - d_gal/dz/arcmin2; 1 - fraction of gals; +2 - angular part for bookkeeping only.
        "STARMASK:", "0",  # Mask over bright stars (Healpix FITS file, write 0 for none).
        "EXTRAP_DIPOLE:", "1",  # If dipole not specified by input Cls; 0 - set dipoles to zero. 1 - extrapolate.
        "LRANGE:", str(ell_min), str(ell_max),  # Minimum and maximum l for which covariance matrices will be generated.
        "CROP_CL:", "1",  # Transf. lognormal to Gaussian Cl using all available L (0) or up to LMAX above (1).
        "SHEAR_LMAX:", str(ell_max),  # Max. l for kappa->shear computations, should be <=NSIDE, only affects lognormal sims.
        "NSIDE:", str(nside),  # Healpix Nside (Npixels = 12*Nside^2).
        "USE_HEALPIX_WGTS:", "1",  # 0 - Use equal (1.0) weights in map2alm; 1 - Use Healpix weights in map2alm.
        "MINDIAG_FRAC:", "1e-12",  # Null diagonal elements are set to this*(smallest diagonal element in all ells).
        "BADCORR_FRAC:", "0.001",  # Fraction added to variances if aux. Cov. matrices lead to |correlation|>1.
        "REGULARIZE_METHOD:", "1",  # Correcting Cov. matrices for pos. def. 0-None; 1-Min. chi-sq; 2-Sampler with steps.
        "NEW_EVAL:", "1e-18",  # Eigenvalue to replace negative ones (only for method 1).
        "REGULARIZE_STEP:", "0.0001",  # Step size for successive approximations to pos. def.ness (only for method 2).
        "REG_MAXSTEPS:", "1000",  # Maximum number of iterations (only for method 2).
        "ADD_FRAC:", "1e-10",  # If Eigenvalues>0 & Cholesky fail, add this*(smallest diagonal element) to diagonal.
        "ZSEARCH_TOL:", "0.0001",  # Precision for finding radial selection function maximum location.
        "EXIT_AT:", "0",  # Write name of last output to be created (program will stop there). 0 for full run.
        "FITS2TGA:", "0",  # 0 - Only FITS; 1 - FITS and TGA; 2 - Only TGA.
        "USE_UNSEEN:", "1",  # Fill masked regions of output Healpix maps with healpy's UNSEEN (1) or zero (0).
        "LRANGE_OUT:", str(ell_min), str(ell_max),  # Inclusive l interval for alm and Cls output. Irrelevant if there is no such output.
        "MMAX_OUT:", "-1",  # Truncate m for alm output at this value. Set it to <0 for m<=l.
        "ANGULAR_COORD:", "2",  # 0 - ThetaPhi in radians 1 - ThetaPhi in degrees 2 - RADEC in degrees.
        "DENS2KAPPA:", "0",  # Integrate (1) or not (0) density to obtain convergence.
        "FLIST_OUT:", "0",
        "SMOOTH_CL_PREFIX:", "0",
        "XIOUT_PREFIX:", "0",
        "GXIOUT_PREFIX:", "0",
        "GCLOUT_PREFIX:", "0",
        "COVL_PREFIX:", "0",
        "REG_COVL_PREFIX:", "0",
        "REG_CL_PREFIX:", "0",
        "CHOLESKY_PREFIX:", "0",
        "AUXALM_OUT:", "0",
        "RECOVAUXCLS_OUT:", "0",
        "AUXMAP_OUT:", "0",
        "DENS2KAPPA_STAT:", "0",
        "MAP_OUT:", "0",
        "MAPFITS_PREFIX:", flask_output_dir + "map-",
        "RECOVALM_OUT:", "0",
        "RECOVCLS_OUT:", "0",
        "SHEAR_ALM_PREFIX:", "0",
        "SHEAR_FITS_PREFIX:", flask_output_dir + "kappa-gamma-",
        "SHEAR_MAP_OUT:", "0",
        "MAPWER_OUT:", "0",
        "MAPWERFITS_PREFIX:", "0",
        "ELLIP_MAP_OUT:", "0",
        "ELLIPFITS_PREFIX:", "0",
        "CATALOG_OUT:", "0"
        ])


if __name__ == '__main__':
    main()
