"""
Script to generate additional 3x2pt field maps by linear interpolation if there are redshift sample points that fall
between where Flask generates maps for (i.e. the no. z bins used to generate maps with Flask is less than the total
number of redshift sample points). Repeated over a given number of realisations/iterations.
This was extremely painful to code so...you're welcome lol.
"""

import os
import sys
import h5py
import configparser
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy import interpolate


def interp_config(pipeline_variables_path):

    """
    Set up a config dictionary to execute the map interpolation based on the catalogue simulation inputs

    Parameters
    ----------
    pipeline_variables_path (str):  Path to location of set_variables_cat.ini file

    Returns
    -------
    Dictionary of pipeline, 3x2pt and redshift parameters
    """

    config = configparser.ConfigParser()
    config.read(pipeline_variables_path)

    zmin = float(config['redshift_distribution']['ZMIN'])
    zmax = float(config['redshift_distribution']['ZMAX'])
    dz = float(config['redshift_distribution']['DZ'])
    nside = int(config['simulation_setup']['NSIDE'])
    nbins = int(config['create_nz']['N_ZBIN'])

    iter_no = int(float(os.environ['ITER_NO']))

    save_dir = str(config['simulation_setup']['SIMULATION_SAVE_DIR'])
    flask_output_dir = save_dir + 'flask/output/iter_{}/'.format(iter_no)

    interp_cluster_path = save_dir + 'flask/interp_maps/clustering/iter_{}/'.format(iter_no)
    interp_conv_path = save_dir + 'flask/interp_maps/convergence/iter_{}/'.format(iter_no)
    interp_shear_path = save_dir + 'flask/interp_maps/shear/iter_{}/'.format(iter_no)

    plot_maps = str(config['simulation_setup']['PLOT_MAPS'])

    z_sample = np.linspace(
        zmin,
        zmax,
        (round((zmax - zmin) / dz)) + 1
    )

    z_sample = z_sample.round(decimals=2)

    zs = np.loadtxt(save_dir + 'zs_medians.txt')
    zs = zs.round(decimals=2)

    # Prepare config dictionary
    config_dict = {
        'save_dir': save_dir,
        'flask_output_dir': flask_output_dir,
        'interp_cluster_path': interp_cluster_path,
        'interp_conv_path': interp_conv_path,
        'interp_shear_path': interp_shear_path,
        'zmin': zmin,
        'zmax': zmax,
        'dz': dz,
        'z_sample': z_sample,
        'zs': zs,
        'nside': nside,
        'nbins': nbins,
        'iter_no': iter_no,
        'plot_maps': plot_maps
    }

    return config_dict


def load_map_slices(config, slice_i, slice_j, field_type):

    """
    Load a given pair of field maps of a given type from the Flask output to prepare for interpolation between the two
    maps' pixel values

    Parameters
    ----------
    config (dict):  Dictionary of the pipeline parameters used for the catalogue simulation
    slice_i (int):  Redshift-space ID of the first map to load in for pair-wise pixel interpolation
    slice_j (int):  Redshift-space ID of the second map to load in for pair-wise pixel interpolation
    field_type (str):   The given 3x2pt field to interpolate field values for. Must be one of 'Clustering',
                        'Convergence', 'Shear_y1', or 'Shear_y2'

    Returns
    -------
    Array containing the map data for the two fields between which to interpolate pixel values
    """

    flask_output_dir = config['flask_output_dir']

    accepted_field_types = {'Clustering', 'Convergence', 'Shear_y1', 'Shear_y2'}

    if field_type not in accepted_field_types:
        print('Error! Field Type Not Recognised - Exiting...')
        sys.exit()

    if field_type == 'Clustering':

        dat = np.transpose(np.array([
            hp.read_map(flask_output_dir + 'map-f2z{}.fits'.format(slice_i)),
            hp.read_map(flask_output_dir + 'map-f2z{}.fits'.format(slice_j))]))

    elif field_type == 'Convergence':

        dat = np.transpose(np.array([
            hp.read_map(flask_output_dir + 'map-f1z{}.fits'.format(slice_i)),
            hp.read_map(flask_output_dir + 'map-f1z{}.fits'.format(slice_j))
        ]))

    elif field_type == 'Shear_y1':

        dat = np.transpose(np.array([
            hp.read_map(flask_output_dir + 'kappa-gamma-f1z{}.fits'.format(slice_i), field=1),
            hp.read_map(flask_output_dir + 'kappa-gamma-f1z{}.fits'.format(slice_j), field=1)
        ]))

    else:

        assert field_type == 'Shear_y2'

        dat = np.transpose(np.array([
            hp.read_map(flask_output_dir + 'kappa-gamma-f1z{}.fits'.format(slice_i), field=2),
            hp.read_map(flask_output_dir + 'kappa-gamma-f1z{}.fits'.format(slice_j), field=2)
        ]))

    return dat


def save_interp_map(config, ras, decs, map_name, map_arr, z_at_slice, field_type):

    """
    Save the interpolated field map (Healpix array) of a given type at a given redshift to disk

    Parameters
    ----------
    config (dict):  Config dictionary of pipeline parameters
    ras (arr):      Array of RA values corresponding to each Healpix pixel (index ordered from 0 -> Npix)
    decs (arr):     Array of Dec values corresponding to each Healpix pixel (index ordered from 0 -> Npix)
    map_name (str): Name with which to save interpolated map to disk
    map_arr (str):  The Healpix map array containing the interpolated field values
    z_at_slice (float): Redshift at which the interpolated field map is evaluated
    field_type (str):   Field type of given interpolated map - will dictate which folder on disk to save map into. Must
                        be one of 'Clustering', 'Convergence', or 'Shear'. If type is 'Shear', assumes that the map_arr
                        array is of the form [map_arr_1, map_arr_2], corresponding to the two shear components y1, y2
    """

    interp_cluster_path = config['interp_cluster_path']
    interp_conv_path = config['interp_conv_path']
    interp_shear_path = config['interp_shear_path']
    save_dir = config['save_dir']
    iter_no = config['iter_no']
    plot_maps = config['plot_maps']

    accepted_field_types = {'Clustering', 'Convergence', 'Shear'}

    if field_type not in accepted_field_types:
        print('Error! Field Type Not Recognised - Exiting...')
        sys.exit()

    if field_type == 'Clustering':
        map_arr = map_arr[0]
        map_data = h5py.File(interp_cluster_path + map_name+'.hdf5', 'w')
        map_data.create_dataset("RA", data=ras)
        map_data.create_dataset("Dec", data=decs)
        map_data.create_dataset("Clustering", data=map_arr)
        map_data.close()

        if plot_maps == 'YES':
            plt.figure(figsize=(5, 3.5))
            hp.mollview(map_arr,
                        title="Clustering (Matter Density)")
            plt.title(
                "Redshift z = %.2f" % (z_at_slice)
            )

            plot_dir = save_dir + 'flask/interp_maps/plots/clustering/iter_{}/'.format(iter_no)
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            plt.savefig(plot_dir + map_name + '.png')
            plt.close()

    if field_type == 'Convergence':
        map_arr = map_arr[0]
        map_data = h5py.File(interp_conv_path + map_name + '.hdf5', 'w')
        map_data.create_dataset("RA", data=ras)
        map_data.create_dataset("Dec", data=decs)
        map_data.create_dataset("Convergence", data=map_arr)
        map_data.close()

        if plot_maps == 'YES':
            plt.figure(figsize=(5, 3.5))
            hp.mollview(map_arr,
                        title=r'$\kappa$' + " Convergence")
            plt.title(
                "Redshift z = %.2f" % (z_at_slice)
            )

            plot_dir = save_dir + 'flask/interp_maps/plots/convergence/iter_{}/'.format(iter_no)
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            plt.savefig(plot_dir + map_name + '.png')
            plt.close()

    if field_type == 'Shear':

        y1_arr = map_arr[0]
        y2_arr = map_arr[1]
        map_data = h5py.File(interp_shear_path + map_name + '.hdf5', 'w')
        map_data.create_dataset("RA", data=ras)
        map_data.create_dataset("Dec", data=decs)
        map_data.create_dataset("Shear_y1", data=y1_arr)
        map_data.create_dataset("Shear_y2", data=y2_arr)
        map_data.close()

        if plot_maps == 'YES':
            plt.figure(figsize=(10, 3.5))
            hp.mollview(y1_arr,
                        sub=(121),
                        title=r'$\gamma$' + "1 Shear ['Q_POLARISATION']",
                        cmap='ocean')
            hp.mollview(y2_arr,
                        sub=(122),
                        title=r'$\gamma$' + "2 Shear ['U_POLARISATION']",
                        cmap='Purples')
            plt.suptitle(
                "Redshift z = %.2f" % (z_at_slice)
            )
            plt.tight_layout()

            plot_dir = save_dir + 'flask/interp_maps/plots/shear/iter_{}/'.format(iter_no)
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            plt.savefig(plot_dir + map_name + '.png')
            plt.close()


def setup_interpolation(config, field, pair_type, pair_id=0):

    """
    Set up the interpolation functions depending on where the maps are located in redshift-space. If the pair of maps
    to interpolate between are at the wings of the redshift distribution we perform a linear extrapolation to fill in
    additional redshift sample points. If the pair of maps are anywhere in the 'middle' of the redshift distribution
    (i.e.) not the pairs on either end, then set up a linear interpolation.

    Parameters
    ----------
    config (dict):  Config dict on pipeline parameters
    field (str):    Given field type - used to find directory to load Flask compiled maps and set up interpolation
                    functions
    pair_type (str):    'First', 'Middle', or 'Last' - pair type of Flask maps to set up either interpolation or
                        extrapolation
    pair_id (float):    The 'ID' corresponding to where the maps are in redshift-space

    Returns
    -------

    """

    zs = config['zs']
    zmin = config['zmin']
    zmax = config['zmax']
    dz = config['dz']
    nbins = config['nbins']

    if pair_type == 'First':

        itp = interpolate.interp1d(
            [zs[0], zs[1]],
            load_map_slices(config=config, slice_i=1, slice_j=2, field_type=field),
            kind='linear',
            fill_value='extrapolate')

        interp_dat = np.transpose(itp(np.linspace(zmin, zs[0], round((zs[0]-zmin)/dz), endpoint=False)))

    elif pair_type == 'Middle':

        itp = interpolate.interp1d(
            [zs[pair_id], zs[pair_id+1]],
            load_map_slices(config=config, slice_i=pair_id+1, slice_j=pair_id+2, field_type=field),
            kind='linear')

        interp_dat = np.transpose(
            itp(np.linspace(zs[pair_id], zs[pair_id+1], round((zs[pair_id+1]-zs[pair_id])/dz), endpoint=False))
        )

    else:
        assert pair_type == 'Last'

        itp = interpolate.interp1d(
            [zs[-2], zs[-1]],
            load_map_slices(config=config, slice_i=nbins - 1, slice_j=nbins, field_type=field),
            kind='linear',
            fill_value='extrapolate')

        interp_dat = np.transpose(itp(np.linspace(zs[-1], zmax, round((zmax - zs[-1]) / dz) + 1)))

    return interp_dat


def execute_interpolation(config, ras, decs, field):

    """
    Execute the map interpolation. First determine what field to load, then iterate through pairs of Flask maps in
    the redshift sample to generate interpolated field maps and then save to disk.

    Parameters
    ----------
    config (dict):  Dictionary of pipeline config parameters
    ras (arr):      Array of RA values corresponding to each Healpix pixel (following Healpix indexing from 0 ->
                    Npix)
    decs (arr):     Array of Dec values corresponding to each Healpix pixel (following Healpix indexing from 0 ->
                    Npix)
    field (str):    Given field type - used to find and load Flask-generated maps
    """

    z_sample = config['z_sample']
    nbins = config['nbins']

    if field == 'Clustering' or field == 'Convergence':

        counter = 0

        interp_dat_first = setup_interpolation(config=config, field=field, pair_type='First')
        for i in range(len(interp_dat_first)):
            counter += 1
            save_interp_map(
                config=config,
                ras=ras,
                decs=decs,
                map_name='map_zslice_{}'.format(counter),
                map_arr=[interp_dat_first[i]],
                z_at_slice=z_sample[i],
                field_type=field)

        for i in range(nbins-1):

            interp_dat = setup_interpolation(config=config, field=field, pair_type='Middle', pair_id=i)

            for j in range(len(interp_dat)):
                counter += 1
                save_interp_map(
                    config=config,
                    ras=ras,
                    decs=decs,
                    map_name='map_zslice_{}'.format(counter),
                    map_arr=[interp_dat[j]],
                    z_at_slice=z_sample[counter-1],
                    field_type=field)

        interp_dat_last = setup_interpolation(config=config, field=field, pair_type='Last')

        for i in range(len(interp_dat_last)):
            counter += 1
            save_interp_map(
                config=config,
                ras=ras,
                decs=decs,
                map_name='map_zslice_{}'.format(counter),
                map_arr=[interp_dat_last[i]],
                z_at_slice=z_sample[counter-1],
                field_type=field)

    if field == 'Shear':

        counter = 0

        interp_dat_first_y1 = setup_interpolation(config=config, field='Shear_y1', pair_type='First')
        interp_dat_first_y2 = setup_interpolation(config=config, field='Shear_y2', pair_type='First')

        for i in range(len(interp_dat_first_y1)):
            counter += 1
            save_interp_map(
                config=config,
                ras=ras,
                decs=decs,
                map_name='map_zslice_{}'.format(counter),
                map_arr=[interp_dat_first_y1[i], interp_dat_first_y2[i]],
                z_at_slice=z_sample[i],
                field_type=field)

        for i in range(nbins - 1):
            interp_dat_y1 = setup_interpolation(config=config, field='Shear_y1', pair_type='Middle', pair_id=i)
            interp_dat_y2 = setup_interpolation(config=config, field='Shear_y2', pair_type='Middle', pair_id=i)

            for j in range(len(interp_dat_y1)):
                counter += 1
                save_interp_map(
                    config=config,
                    ras=ras,
                    decs=decs,
                    map_name='map_zslice_{}'.format(counter),
                    map_arr=[interp_dat_y1[j], interp_dat_y2[j]],
                    z_at_slice=z_sample[counter - 1],
                    field_type=field)

        interp_dat_last_y1 = setup_interpolation(config=config, field='Shear_y1', pair_type='Last')
        interp_dat_last_y2 = setup_interpolation(config=config, field='Shear_y2', pair_type='Last')

        for i in range(len(interp_dat_last_y1)):
            counter += 1
            save_interp_map(
                config=config,
                ras=ras,
                decs=decs,
                map_name='map_zslice_{}'.format(counter),
                map_arr=[interp_dat_last_y1[i], interp_dat_last_y2[i]],
                z_at_slice=z_sample[counter - 1],
                field_type=field)


def main():

    """
    Main function to run the interpolation routine. Load in the pipeline variables to prep into a config dictionary,
    then execute the interpolation for each field type.
    """

    pipeline_variables_path = os.environ['PIPELINE_VARIABLES_PATH']
    interp_config_dict = interp_config(pipeline_variables_path=pipeline_variables_path)

    nside = interp_config_dict['nside']
    interp_cluster_path = interp_config_dict['interp_cluster_path']
    interp_conv_path = interp_config_dict['interp_conv_path']
    interp_shear_path = interp_config_dict['interp_shear_path']
    plot_maps = interp_config_dict['plot_maps']

    npix = hp.nside2npix(nside)
    ras = hp.pix2ang(nside, np.arange(0, npix), lonlat=True)[0]
    decs = hp.pix2ang(nside, np.arange(0, npix), lonlat=True)[1]

    # create directory to store interpolated maps
    if not os.path.exists(interp_cluster_path):
        os.makedirs(interp_cluster_path)

    if not os.path.exists(interp_conv_path):
        os.makedirs(interp_conv_path)

    if not os.path.exists(interp_shear_path):
        os.makedirs(interp_shear_path)

    if plot_maps == 'YES':
        print('PLOTTING INTERPOLATED MAPS')

    else:
        print('NOT PLOTTING INTERPOLATED MAPS')

    execute_interpolation(config=interp_config_dict, ras=ras, decs=decs, field='Clustering')
    execute_interpolation(config=interp_config_dict, ras=ras, decs=decs, field='Convergence')
    execute_interpolation(config=interp_config_dict, ras=ras, decs=decs, field='Shear')


if __name__ == '__main__':
    main()
