"""
Create a tomographic n(z) based on measurement parameters supplied by 'set_variables_3x2pt_measurement.ini'. First, an
array of redshift boundary values for each bin is created and saved to disk, then the n(z) is measured using these
boundaries from the simulated catalogues.
"""

import os
import sys
import h5py
import configparser
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from multiprocessing import Process

def nz_fromsim_config(pipeline_variables_path):

    """
    Set up a config dictionary to generate an n(z) distribution as measured from the simulations

    Parameters
    ----------
    pipeline_variables_path (str):  Path to location of pipeline variables file ('set_variables_3x2pt_measurement.ini')

    Returns
    -------
    Dictionary of n(z) parameters
    """

    config = configparser.ConfigParser()
    config.read(pipeline_variables_path)

    zmin = float(config['create_nz']['ZMIN'])
    zmax = float(config['create_nz']['ZMAX'])

    # Precision/step-size of z-range that is sampled over.
    dz = float(config['create_nz']['DZ'])

    nbins = int(float(config['create_nz']['N_ZBIN']))
    bin_type = str(config['create_nz']['ZBIN_TYPE'])

    nz_table_filename = str(config['create_nz']['MEASURED_NZ_TABLE_NAME'])

    save_dir = str(config['measurement_setup']['MEASUREMENT_SAVE_DIR'])
    catalogue_dir = str(config['measurement_setup']['CATALOGUE_DIR'])
    realisations = int(float(config['measurement_setup']['REALISATIONS']))

    sigma_phot = str(config['noise_cls']['SIGMA_PHOT'])
    sigma_shear = str(config['noise_cls']['SIGMA_SHEAR'])

    # Prepare config dictionary
    config_dict = {
        'zmin': zmin,
        'zmax': zmax,
        'dz': dz,
        'nbins': nbins,
        'bin_type': bin_type,
        'nz_table_filename': nz_table_filename,
        'save_dir': save_dir,
        'catalogue_dir': catalogue_dir,
        'realisations': realisations,
        'sigma_phot': sigma_phot,
        'sigma_shear': sigma_shear
    }

    return config_dict


def create_zbin_boundaries(config_dict):

    """
    Create a table of the redshift boundaries used for binning the galaxies in the simulated catalogues for the 3x2pt
    analysis, which is then saved to disk.

    Parameters
    ----------

    config_dict (dict): Dictionary of pipeline and redshift distribution parameters used to generate the bin boundaries
                        and overall n(z)

    Returns
    -------
    Array of the redshift bin boundaries evaluated for the given number of bins + binning configuration.
    """

    zmin = config_dict['zmin']
    zmax = config_dict['zmax']
    dz = config_dict['dz']
    nbins = config_dict['nbins']
    bin_type = config_dict['bin_type']
    save_dir = config_dict['save_dir']
    catalogue_dir = config_dict['catalogue_dir']

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    z_boundaries_filename = 'z_boundaries.txt'

    if bin_type == 'EQUI_Z':
        z_boundaries_low = np.linspace(zmin, zmax, nbins + 1)
        z_boundaries_mid = z_boundaries_low + (((zmax - zmin) / nbins) / 2)
        z_boundaries_high = z_boundaries_mid + (((zmax - zmin) / nbins) / 2)

        z_boundaries = [z_boundaries_low, z_boundaries_mid, z_boundaries_high]
        np.savetxt(save_dir + z_boundaries_filename,
                   np.transpose(z_boundaries),
                   fmt=['%.2f', '%.2f', '%.2f'])

    elif bin_type == 'EQUI_POP':

        # Need to generate a rnd_sample from the measured n(z), i.e. n(z)*z for each z

        mock_cat_filename = 'Raw_Galaxy_Sample.hdf5'
        mock_cat = catalogue_dir + mock_cat_filename
        with h5py.File(mock_cat, "r") as f:
            rnd_sample = f['Redshift_z'][()]

        # rnd_sample = np.load(save_dir)  # Placeholder for now!

        rnd_sample = np.round(rnd_sample, 2)
        sorted_sample = np.sort(rnd_sample)
        split_sorted_sample = np.array_split(sorted_sample, nbins)
        z_boundaries_low = [zmin]
        z_boundaries_high = []
        for i in range(nbins):
            z_boundaries_low.append(split_sorted_sample[i][-1])
            z_boundaries_high.append(split_sorted_sample[i][-1])
        z_boundaries_high.append(z_boundaries_high[-1] + dz)
        z_boundaries_mid = []
        for i in range(len(z_boundaries_low)):
            z_boundaries_mid.append(round(np.mean([z_boundaries_low[i], z_boundaries_high[i]]), 2))

        z_boundaries = [z_boundaries_low, z_boundaries_mid, z_boundaries_high]
        np.savetxt(save_dir + z_boundaries_filename,
                   np.transpose(z_boundaries),
                   fmt=['%.2f', '%.2f', '%.2f'])

    elif bin_type == 'EQUI_D':
        # we need to go back to the directory of the simulation and into the cosmosis/distances file for the
        # comoving distance as a function of z. Then cut out the range that corresponds to the z_range of observation
        # then define equally spaced boundaries in d-space and take the corresponding z boundaries

        z_distances = np.loadtxt(catalogue_dir + 'cosmosis/distances/z.txt')
        d_m = np.loadtxt(catalogue_dir + 'cosmosis/distances/d_m.txt')
        zmin_id = np.where((z_distances == zmin))[0][0]
        zmax_id = np.where((z_distances == zmax))[0][0]
        #print(zmin_id, zmax_id)
        d_m_observed = d_m[zmin_id:zmax_id+1]
        z_observed = z_distances[zmin_id:zmax_id+1]
        d_m_range = d_m_observed[-1]-d_m_observed[0]
        d_m_separation = d_m_range/nbins
        z_boundaries_low = [zmin]
        z_boundaries_high = []

        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx

        for i in range(nbins):
            obs_id = find_nearest(d_m_observed, d_m_observed[0] + (d_m_separation*(i+1)))
            #print(obs_id)
            z_boundaries_low.append(z_observed[obs_id])
            z_boundaries_high.append(z_observed[obs_id])
        z_boundaries_high.append(z_boundaries_high[-1] + dz)

        z_boundaries_mid = []

        for i in range(len(z_boundaries_low)):
            z_boundaries_mid.append(round(np.mean([z_boundaries_low[i], z_boundaries_high[i]]), 2))

        z_boundaries = [z_boundaries_low, z_boundaries_mid, z_boundaries_high]

        np.savetxt(save_dir + z_boundaries_filename,
                   np.transpose(z_boundaries),
                   fmt=['%.2f', '%.2f', '%.2f'])

    else:
        print(bin_type)
        print("Bin Type Not Recognised! Must be 'EQUI_Z', 'EQUI_POP', or 'EQUI_D'")
        sys.exit()

    return np.asarray(z_boundaries)


def main():

    """
    Generate the n(z) measured from the simulated catalogues. First set up the config dictionary, then create the
    bin boundaries array for the chosen tomogaphy, then save n(z) to disk and plot.
    """

    pipeline_variables_path = os.environ['PIPELINE_VARIABLES_PATH']
    # Create 'Observed Redshift'
    config_dict = nz_fromsim_config(pipeline_variables_path=pipeline_variables_path)
    z_boundaries = create_zbin_boundaries(config_dict=config_dict)


if __name__ == '__main__':
    main()
