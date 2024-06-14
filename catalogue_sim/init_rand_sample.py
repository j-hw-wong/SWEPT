"""
Script to generate a random sample of N galaxies that follows a given p(z) distribution. Parameters are read in from
the 'set_variables_cat.ini' file, and the sample of galaxies is saved as 'Raw_Galaxy_Sample.hdf5' on disk.
"""

import os
import h5py
import configparser
import numpy as np
from random import choices


def pz_config(pipeline_variables_path):

    """
    Set up a config dictionary of cosmology/redshift parameters to generate a sample of galaxies

    Parameters
    ----------
    pipeline_variables_path (str):  Path to the 'set_variables_cat.ini' parameters file that exists within pipeline
                                    folder

    Returns
    -------
    Dictionary of config parameters for initialisation of random galaxies
    """

    config = configparser.ConfigParser()
    config.read(pipeline_variables_path)

    # Constants for the galaxy n(z) probability distribution
    z0 = float(config['redshift_distribution']['Z0'])
    beta = float(config['redshift_distribution']['BETA'])

    # Set z-range to simulate over
    zmin = float(config['redshift_distribution']['ZMIN'])
    zmax = float(config['redshift_distribution']['ZMAX'])

    # Precision/step-size of z-range that is sampled over.
    dz = float(config['redshift_distribution']['DZ'])

    # No. galaxies to simulate
    sample_points = int(float(config['redshift_distribution']['NGAL']))

    save_dir = str(config['simulation_setup']['SIMULATION_SAVE_DIR'])

    # Prepare config dictionary
    config_dict = {
        'z0': z0,
        'beta': beta,
        'zmin': zmin,
        'zmax': zmax,
        'dz': dz,
        'sample_points': sample_points,
        'save_dir': save_dir
    }

    return config_dict


def pz(z, z0, beta):

    """
    Takes as input an array of galaxy redshift values and generates a standard probability distribution

    Parameters
    ----------
    z (array):      Redshift values with which to return a probability distribution
    z0 (float):     Functional constant to normalise the redshift
    beta (float):   Exponential constant for redshift distribution

    Returns
    -------
    Array of the probability values at the given redshifts
    """

    return ((z/z0)**2)*np.exp(-1*((z/z0)**beta))


def init_nz(config_dict):

    """
    Generate a random sample of N galaxies that follows the probability distribution pz and save to disk

    Parameters
    ----------
    config_dict (dict): Dictionary of config parameters set up in pz_config

    """

    zmin = config_dict['zmin']
    zmax = config_dict['zmax']
    dz = config_dict['dz']

    z0 = config_dict['z0']
    beta = config_dict['beta']

    sample_points = config_dict['sample_points']

    save_dir = config_dict['save_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    z_sample = np.linspace(
        zmin,
        zmax,
        (round((zmax-zmin)/dz))+1
    )

    # Select n=sample_points number of galaxies randomly from p(z) distribution
    rnd_sample = choices(z_sample[0:-1], pz(z_sample[0:-1], z0=z0, beta=beta), k=sample_points)
    rnd_sample = np.round(np.asarray(rnd_sample), 2)

    # Filename to save raw sample
    mock_cat_filename = 'Raw_Galaxy_Sample.hdf5'

    with h5py.File(save_dir + mock_cat_filename, 'w') as f:
        f.create_dataset("Redshift_z", data=rnd_sample)


def main():

    """
    Generate the galaxy sample by reading in the pipeline variables file as environment variable, then setting up the
    config dictionary and initialising the n(z)
    """

    pipeline_variables_path = os.environ['PIPELINE_VARIABLES_PATH']
    pz_config_dict = pz_config(pipeline_variables_path=pipeline_variables_path)
    init_nz(config_dict=pz_config_dict)


if __name__ == '__main__':
    main()
