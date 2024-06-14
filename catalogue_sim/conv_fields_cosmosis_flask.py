"""
Convert the 3x2pt data calculated from CosmoSIS into the correct file + field conventions required for the map
generation by Flask
"""

import os
import configparser
import numpy as np


def conversion_config(pipeline_variables_path):

    """
    Set up a config dictionary to execute the CosmoSIS-Flask file conversion based on pipeline parameters
    specified in a given input variables file

    Parameters
    ----------
    pipeline_variables_path (str):  Path to location of set_variables_cat.ini file

    Returns
    -------
    Dictionary of pipeline and file conversion parameters
    """

    config = configparser.ConfigParser()
    config.read(pipeline_variables_path)

    nbins = int(float(config['create_nz']['N_ZBIN']))
    bins = np.arange(1, nbins + 1, 1)

    save_dir = str(config['simulation_setup']['SIMULATION_SAVE_DIR'])

    z_boundaries_filename = 'z_boundaries.txt'
    z_boundaries = np.loadtxt(save_dir + z_boundaries_filename)
    z_boundary_columns = np.transpose(z_boundaries)
    z_boundaries_low = z_boundary_columns[0][0:-1]
    z_boundaries_mid = z_boundary_columns[1][0:-1]
    z_boundaries_high = z_boundary_columns[2][0:-1]

    # Prepare config dictionary
    config_dict = {
        'nbins': nbins,
        'bins': bins,
        'save_dir': save_dir,
        'z_boundaries_low': z_boundaries_low,
        'z_boundaries_mid': z_boundaries_mid,
        'z_boundaries_high': z_boundaries_high
    }

    return config_dict


def open_data(data_file):

    """
    Convenience function to open data (in CosmoSIS format) and store into array

    Parameters
    ----------
    data_file (str):    Path to data file

    Returns
    -------
    Array of data stored on disk
    """

    data_arr = []

    with open(data_file) as f:
        for line in f:
            column = line.split()
            if not line.startswith('#'):
                data_i = float(column[0])
                data_arr.append(data_i)

    return data_arr


def normalise_power(raw_cls, raw_ells):

    """
    Convenience function to turn CosmoSIS output data (raw Cl power spectra) into normalised power spectra for input
    into Flask

    Parameters
    ----------
    raw_cls (arr):  Array of raw Cls
    raw_ells (arr): Array of raw ells

    Returns
    -------
    Array of Normalised Cls
    """

    cl_normalised = []
    for k in range(len(raw_cls)):
        cl_normalised_i = (raw_cls[k] * raw_ells[k] * (raw_ells[k] + 1.0)) / (2 * np.pi)
        cl_normalised.append(cl_normalised_i)

    return cl_normalised


def execute(config_dict):

    """
    Convert the 3x2pt data files output from CosmoSIS into the correct field + naming conventions for Flask

    Parameters
    ----------
    config_dict (dict): Dictionary of pipeline and field parameters for the 3x2pt simulation
    """

    save_dir = config_dict['save_dir']
    nbins = config_dict['nbins']
    bins = config_dict['bins']
    z_boundaries_low = config_dict['z_boundaries_low']
    z_boundaries_mid = config_dict['z_boundaries_mid']
    z_boundaries_high = config_dict['z_boundaries_high']

    flask_data_dir = save_dir + 'flask/data/'
    cosmosis_data_dir = save_dir + 'cosmosis/'

    if not os.path.exists(flask_data_dir):
        os.makedirs(flask_data_dir)

    for i in bins:

        for j in bins:

            ell_file = cosmosis_data_dir + 'galaxy_shear_cl/ell.txt'
            ell = open_data(ell_file)

            gal_shear_txt_file = cosmosis_data_dir + 'galaxy_shear_cl/bin_{}_{}.txt'.format(i, j)
            gal_shear_cl = open_data(gal_shear_txt_file)

            gal_shear_cl_normalised = normalise_power(gal_shear_cl, ell)

            gal_shear_file_name = '/Cl-f2z{}f1z{}.dat'.format(i, j)
            gal_shear_save_file_name = flask_data_dir + gal_shear_file_name
            np.savetxt(gal_shear_save_file_name, np.transpose([ell, gal_shear_cl]), fmt='%.18f')

            if i >= j:
                ell_file = cosmosis_data_dir + 'shear_cl/ell.txt'
                ell = open_data(ell_file)

                shear_txt_file = cosmosis_data_dir + 'shear_cl/bin_{}_{}.txt'.format(i, j)
                shear_cl = open_data(shear_txt_file)

                shear_cl_normalised = normalise_power(shear_cl, ell)

                shear_file_name = '/Cl-f1z{}f1z{}.dat'.format(i, j)
                shear_save_file_name = flask_data_dir + shear_file_name
                np.savetxt(shear_save_file_name, np.transpose([ell, shear_cl]), fmt='%.18f')

                gal_txt_file = cosmosis_data_dir + 'galaxy_cl/bin_{}_{}.txt'.format(i, j)
                gal_cl = open_data(gal_txt_file)

                gal_cl_normalised = normalise_power(gal_cl, ell)

                gal_file_name = '/Cl-f2z{}f2z{}.dat'.format(i, j)
                gal_save_file_name = flask_data_dir + gal_file_name
                np.savetxt(gal_save_file_name, np.transpose([ell, gal_cl]), fmt='%.18f')

    gal_field = 1
    wl_field = 2

    field_nos = np.zeros(nbins)

    gal_field_nos = field_nos + 2
    wl_field_nos = field_nos + 1

    z_bin_number = bins
    mean = np.zeros(nbins)
    shift = np.zeros(nbins)
    shift = shift + 1

    field_type = np.zeros(nbins)
    gal_field_type = field_type + gal_field
    wl_field_type = field_type + wl_field

    field_info_gal = [
        gal_field_nos,
        z_bin_number,
        mean,
        shift,
        gal_field_type,
        z_boundaries_low,
        z_boundaries_high
    ]

    field_info_wl = [
        wl_field_nos,
        z_bin_number,
        mean,
        shift,
        wl_field_type,
        z_boundaries_low,
        z_boundaries_high
    ]

    field_info_3x2pt = np.concatenate((field_info_wl, field_info_gal), axis=1)

    field_info_3x2pt_filename = 'field_info_3x2pt.dat'
    field_info_wl_filename = 'field_info_wl.dat'
    field_info_gal_filename = 'field_info_gal.dat'

    np.savetxt(
        flask_data_dir + field_info_3x2pt_filename,
        np.transpose(field_info_3x2pt),
        fmt=['%6i', '%6i', '%10.4f', '%10.4f', '%6i', '%10.4f', '%10.4f'],
        header='Field number, z bin number, mean, shift, field type, zmin, zmax\nTypes: 1-galaxies 2-lensing\n'
    )

    np.savetxt(
        flask_data_dir + field_info_wl_filename,
        np.transpose(field_info_wl),
        fmt=['%6i', '%6i', '%10.4f', '%10.4f', '%6i', '%10.4f', '%10.4f'],
        header='Field number, z bin number, mean, shift, field type, zmin, zmax\nTypes: 1-galaxies 2-lensing\n'
    )

    np.savetxt(
        flask_data_dir + field_info_gal_filename,
        np.transpose(field_info_gal),
        fmt=['%6i', '%6i', '%10.4f', '%10.4f', '%6i', '%10.4f', '%10.4f'],
        header='Field number, z bin number, mean, shift, field type, zmin, zmax\nTypes: 1-galaxies 2-lensing\n'
    )


def main():

    """
    Generate and save the Flask 3x2pt field data files by reading in the pipeline variables file as environment
    variable, then setting up the config dictionary and converting the CosmoSIS field information saved on disk
    """

    pipeline_variables_path = os.environ['PIPELINE_VARIABLES_PATH']
    conversion_config_dict = conversion_config(pipeline_variables_path=pipeline_variables_path)
    execute(config_dict=conversion_config_dict)


if __name__ == '__main__':
    main()
