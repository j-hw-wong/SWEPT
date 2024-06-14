"""
Script to average the measured 3x2pt Pseudo Cls measured over multiple realisations by measure_cat_3x2pt_pcls.py.
Specifies in particular the final ell range that will be saved and used for any analysis
"""

import os
import linecache
import statistics
import configparser
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem


def av_cls_config(pipeline_variables_path):

    """
    Create a dictionary of parameters that will be useful to calculate average power spectra.

    Parameters
    ----------
    pipeline_variables_path (str):  Path to location of pipeline variables file ('set_variables_3x2pt_measurement.ini')

    Returns
    -------
    Dictionary of parameters used by this script to measure average 3x2pt power spectra
    """

    config = configparser.ConfigParser()
    config.read(pipeline_variables_path)

    save_dir = str(config['measurement_setup']['MEASUREMENT_SAVE_DIR'])
    nside = int(config['measurement_setup']['NSIDE'])
    realisations = int(config['measurement_setup']['realisations'])
    pcl_lmin_out = 0
    pcl_lmax_out = int(float(config['measurement_setup']['INPUT_ELL_MAX']))

    nbins = int(config['create_nz']['N_ZBIN'])

    nz_table_filename = str(config['create_nz']['MEASURED_NZ_TABLE_NAME'])

    # Prepare config dictionary
    config_dict = {
        'nside': nside,
        'nbins': nbins,
        'pcl_lmin_out': pcl_lmin_out,
        'pcl_lmax_out': pcl_lmax_out,
        'save_dir': save_dir,
        'realisations': realisations,
        'nz_table_filename': nz_table_filename
    }

    return config_dict


def calc_av_cls(cl_dir, ell_min, ell_max, bin_i, bin_j, realisations):

    """
    Calculate average Pseudo-Cls for all tomographic bins and all realisations that is stored in a given directory.

    Parameters
    ----------
    cl_dir (str):   Path to where the 'raw' power spectra (measured from measure_cat_3x2pt_pcls.py) are stored. There
                    should be subdirectories for each realisation within this directory.
    ell_min (float):    Output minimum ell to save the power spectra on disk
    ell_max (float):    Output maximum ell to save the power spectra on disk
    bin_i (float):  Tomographic bin id number of the first field
    bin_j (float):  Tomographic bin id number of the second field
    realisations:   Number of realisations that the power spectra are averaged over

    Returns
    -------
    txt files of the averaged 3x2pt power spectra for each tomographic component. Saved to the cl_dir path.
    """

    cls = []
    ell = np.arange(ell_min, ell_max + 1)
    for x in range(len(ell)):
        cl_av = []
        for y in range(realisations):
            cl_file = cl_dir + 'iter_{}/bin_{}_{}.txt'.format(y + 1, bin_i, bin_j)
            a = linecache.getline(
                cl_file,
                x + 1).split()
            cl_av.append(float(a[0]))

        cls.append(statistics.mean(cl_av))

    np.savetxt(cl_dir + 'bin_{}_{}.txt'.format(bin_i, bin_j),
               np.transpose(cls))

    np.savetxt(cl_dir + 'ell.txt', np.transpose(ell))


def calc_stdem_cls(cl_dir, ell_min, ell_max, bin_i, bin_j, realisations):

    """
    Calculate standard error on mean of Pseudo-Cls for all tomographic bins and all realisations that is stored in a
    given directory.

    Parameters
    ----------
    cl_dir (str):   Path to where the 'raw' power spectra (measured from measure_cat_3x2pt_pcls.py) are stored. There
                    should be subdirectories for each realisation within this directory.
    ell_min (float):    Output minimum ell to save the power spectra on disk
    ell_max (float):    Output maximum ell to save the power spectra on disk
    bin_i (float):  Tomographic bin id number of the first field
    bin_j (float):  Tomographic bin id number of the second field
    realisations:   Number of realisations that the power spectra are averaged over

    Returns
    -------
    txt files of the standard error on mean for the 3x2pt power spectra for each tomographic component. Saved to the
    cl_dir path.
    """

    cls_err = []
    ell = np.arange(ell_min, ell_max + 1)

    for x in range(len(ell)):

        cl_av = []

        for y in range(realisations):
            cl_file = cl_dir + 'iter_{}/bin_{}_{}.txt'.format(y + 1, bin_i, bin_j)
            a = linecache.getline(
                cl_file,
                x + 1).split()
            cl_av.append(float(a[0]))

        cls_err.append(sem(cl_av))

    np.savetxt(cl_dir + 'bin_%s_%s_err.txt' % (bin_i, bin_j),
               np.transpose(cls_err))


def calc_av_nz(nz_tables_dir, realisations):

    """
    Function to calculate the average n(z) measured over all realisations.

    Parameters
    ----------
    nz_tables_dir (str):    Path to location that the n(z) tables are stored for each realisation. This is defined in
                            the measure_cat_3x2pt_pcls.py script.
    realisations (float):   Total number of realisations to average the measured n(z).

    Returns
    -------
    Array of the average n(z) measured over all realisations.
    """

    nz_dat = []
    for i in range(realisations):

        nz_dat.append(np.transpose(np.loadtxt(nz_tables_dir + 'nz_iter{}.txt'.format(i+1))))

    return np.mean(np.asarray(nz_dat), axis=0)


def main():

    """
    Main function to execute the averaging measurements.
    """

    # First create a dictionary of parameter used in this script and extract the useful ones
    pipeline_variables_path = os.environ['PIPELINE_VARIABLES_PATH']
    config_dict = av_cls_config(pipeline_variables_path=pipeline_variables_path)

    save_dir = config_dict['save_dir']
    nbins = config_dict['nbins']

    pcl_lmin_out = config_dict['pcl_lmin_out']
    pcl_lmax_out = config_dict['pcl_lmax_out']

    realisations = config_dict['realisations']

    nz_table_filename = config_dict['nz_table_filename']

    noise_cls_dir = save_dir + 'raw_noise_cls/'
    measured_cls_dir = save_dir + 'raw_3x2pt_cls/'

    # Calculate the average n(z) over all realisations, and save to the MEASUREMENT_SAVE_DIR location specified in the
    # pipeline variables set_variables_3x2pt_measurement.ini file

    final_nz_table = calc_av_nz(nz_tables_dir=save_dir+'nz_tables/', realisations=realisations)
    np.savetxt(save_dir + nz_table_filename, np.transpose(final_nz_table))

    # Plot the average n(z)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, ax1 = plt.subplots(figsize=(7.5, 6))

    for i in range(len(final_nz_table) - 1):
        zs = final_nz_table[0]

        obs_nzs = final_nz_table[i + 1]
        obs_nzs = obs_nzs.astype(np.float64)
        obs_nzs[obs_nzs == 0] = np.nan

        ax1.plot(zs, obs_nzs, 'o', linestyle='--',markersize=5, color=colors[i])

    ax1.set_xlabel('Redshift ' r'$z$', fontsize=15, labelpad=10)
    ax1.set_ylabel(r'$n(z)$' ' [No. Galaxies/' r'$dz=0.1$' ']', fontsize=15, labelpad=10)
    ax1.tick_params(axis="both", direction="in")

    ax1.tick_params(right=True, top=True, labelright=False, labeltop=False)
    ax1.tick_params(axis='both', which='major', labelsize=13.5)

    plt.setp(ax1.xaxis.get_majorticklabels(), ha="center")

    plt.savefig(save_dir + 'nz.png')

    # Now calculate the average power spectra for each tomographic 3x2pt component.

    for i in range(nbins):
        for j in range(nbins):

            # First average signal and noise for galaxy-shear

            calc_av_cls(cl_dir=noise_cls_dir + 'galaxy_shear_cl/',
                        ell_min=pcl_lmin_out,
                        ell_max=pcl_lmax_out,
                        bin_i=i + 1,
                        bin_j=j + 1,
                        realisations=realisations)

            calc_av_cls(cl_dir=measured_cls_dir + 'galaxy_shear_cl/',
                        ell_min=pcl_lmin_out,
                        ell_max=pcl_lmax_out,
                        bin_i=i + 1,
                        bin_j=j + 1,
                        realisations=realisations)

            calc_stdem_cls(cl_dir=measured_cls_dir + 'galaxy_shear_cl/',
                           ell_min=pcl_lmin_out,
                           ell_max=pcl_lmax_out,
                           bin_i=i + 1,
                           bin_j=j + 1,
                           realisations=realisations)

            # Now average signal and noise for galaxy component

            if i >= j:
                calc_av_cls(noise_cls_dir + 'galaxy_cl/',
                            ell_min=pcl_lmin_out,
                            ell_max=pcl_lmax_out,
                            bin_i=i + 1,
                            bin_j=j + 1,
                            realisations=realisations)

                calc_av_cls(measured_cls_dir + 'galaxy_cl/',
                            ell_min=pcl_lmin_out,
                            ell_max=pcl_lmax_out,
                            bin_i=i + 1,
                            bin_j=j + 1,
                            realisations=realisations)

                calc_stdem_cls(measured_cls_dir + 'galaxy_cl/',
                               ell_min=pcl_lmin_out,
                               ell_max=pcl_lmax_out,
                               bin_i=i + 1,
                               bin_j=j + 1,
                               realisations=realisations)

                calc_av_cls(noise_cls_dir + 'shear_cl/',
                            ell_min=pcl_lmin_out,
                            ell_max=pcl_lmax_out,
                            bin_i=i + 1,
                            bin_j=j + 1,
                            realisations=realisations)

    # Now average signal and noise for each shear component

    cl_shear_types = ['Cl_TT', 'Cl_EE', 'Cl_EB', 'Cl_BE', 'Cl_BB']

    for shear_type in cl_shear_types:
        for i in range(nbins):
            for j in range(nbins):
                if i >= j:
                    calc_av_cls(measured_cls_dir + 'shear_cl/' + shear_type + '/',
                                ell_min=pcl_lmin_out,
                                ell_max=pcl_lmax_out,
                                bin_i=i + 1,
                                bin_j=j + 1,
                                realisations=realisations)


if __name__ == '__main__':
    main()
