"""
Script to Poisson-sample the matter density field traced by the p(z) redshift distribution at each map slice in
redshift space. Generates a number count of galaxies per Healpix pixel, which is saved as a simple table
'cat_indices.hdf5' on disk to then load and assign shear values to in the following 'compile_cat.py' script.
Repeated over a given number of realisations/iterations.
"""

import os
import h5py
import configparser
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


def poisson_sample_masked_map(expected_numb_dens_map, mask, n_pixels):

    """
    Function to poisson sample a given matter field map by returning an integer number count of galaxies per pixel

    Parameters
    ----------
    expected_numb_dens_map (arr):   Array of the 'expected' number density of galaxies converted from the raw density
                                    field
    mask (arr): Healpix map of the given mask used for observation
    n_pixels (float):   Number of pixels in the given map

    Returns
    -------
    A Healpix map contaning the per pixel integer number density of galaxies. Pixels that fall outside of the mask are
    given the hp.UNSEEN value.
    """

    obs_inds = np.where(mask == 1)[0]
    mask_inds = np.where(mask == 0)[0]

    expected_numb_dens_obs = expected_numb_dens_map[obs_inds]
    rng = np.random.default_rng()
    numb_dens_obs = rng.poisson(lam=expected_numb_dens_obs)

    final_numb_dens_map = np.zeros(n_pixels)
    np.add.at(final_numb_dens_map, obs_inds, numb_dens_obs)
    final_numb_dens_map[mask_inds] = hp.UNSEEN

    return final_numb_dens_map


def execute_poisson_sampling():

    """
    Execute the Poisson sampling routine - load in pipeline parameters from variables file, then apply observation
    mask and Poisson sample the observed pixels. This process essentially assigns each galaxy in the sample a given
    Healpix pixel ID - the location on the sky where it is observed. This table is then saved to disk under the
    'cat_products/cat_indices/' location.
    """

    pipeline_variables_path = os.environ['PIPELINE_VARIABLES_PATH']

    config = configparser.ConfigParser()
    config.read(pipeline_variables_path)

    nside = int(config['simulation_setup']['NSIDE'])
    npix = hp.nside2npix(nside)

    save_dir = str(config['simulation_setup']['SIMULATION_SAVE_DIR'])

    iter_no = int(float(os.environ['ITER_NO']))

    mock_cat_filename = 'Raw_Galaxy_Sample.hdf5'
    mock_cat = save_dir + mock_cat_filename
    with h5py.File(mock_cat, "r") as f:
        rnd_sample = f['Redshift_z'][()]

    rnd_sample = np.round(rnd_sample, 2)

    # Set z-range to simulate over
    zmin = float(config['redshift_distribution']['ZMIN'])
    zmax = float(config['redshift_distribution']['ZMAX'])

    # Precision/step-size of z-range that is sampled over.
    dz = float(config['redshift_distribution']['DZ'])

    z_sample = np.linspace(
        zmin,
        zmax,
        (round((zmax - zmin) / dz)) + 1
    )

    z_sample = z_sample.round(decimals=2)

    map_dir = save_dir + 'flask/interp_maps/clustering/iter_{}/'.format(iter_no)
    mask_path = str(config['simulation_setup']['PATH_TO_MASK'])

    mask_map = hp.read_map(mask_path)
    masked_pixels = np.count_nonzero(mask_map == 0)

    numb_dens_maps = []
    corr_pixels = []

    final_positions = []
    final_redshifts = []

    cats_dir = save_dir + 'cat_products/'
    if not os.path.exists(cats_dir):
        os.makedirs(cats_dir)

    cat_indices_dir = cats_dir + 'cat_indices/'.format(iter_no)
    if not os.path.exists(cat_indices_dir):
        os.makedirs(cat_indices_dir)

    for k in range(len(z_sample)):

        arr = z_sample[k] + np.zeros(np.count_nonzero(rnd_sample == z_sample[k]))

        f = h5py.File(map_dir + 'map_zslice_{}.hdf5'.format(k + 1), 'r')
        raw_field_vals = np.array(f.get('Clustering'))
        f.close()

        # For if you want to Poisson sample a homogeneous map - useful for e.g.
        # debugging, testing/comparing noise levels etc.
        # raw_field_vals = np.zeros(npix)

        norm_field_vals = raw_field_vals

        corrected_pixels = sum(pix < -1 for pix in raw_field_vals[np.where(mask_map == 1)])
        corr_pixels.append(corrected_pixels)

        norm_field_vals[norm_field_vals < -1] = -1

        nbar = len(arr) / (npix - masked_pixels)

        expected_numb_dens = (norm_field_vals + 1) * nbar

        numb_dens_map = poisson_sample_masked_map(expected_numb_dens, mask_map, npix)

        if str(config['simulation_setup']['PLOT_MAPS']) == 'YES':

            numb_dens_maps_dir = cats_dir + 'numb_dens_maps/iter_{}/'.format(iter_no)
            if not os.path.exists(numb_dens_maps_dir):
                os.makedirs(numb_dens_maps_dir)

            hp.mollview(numb_dens_map, title='Integer Number Density z=%s' % (z_sample[k]))
            plt.savefig(numb_dens_maps_dir + 'map_zslice_%s.png' % (k + 1))

        ids = []

        for n in range(len(numb_dens_map)):
            pixel_occupation = numb_dens_map[n]
            if pixel_occupation != hp.UNSEEN:
                ids.extend([n] * int(numb_dens_map[n]))

        ids = np.asarray(ids)
        z_vals = np.zeros(len(ids))
        z_vals = z_vals + z_sample[k]

        final_positions.append(ids)
        final_redshifts.append(z_vals)

    final_positions = [item for sublist in final_positions for item in sublist]
    final_redshifts = [item for sublist in final_redshifts for item in sublist]

    final_positions = np.asarray(final_positions)
    final_redshifts = np.asarray(final_redshifts)

    # For .hdf5 format
    cat = h5py.File(cat_indices_dir + 'cat_indices_iter_{}.hdf5'.format(iter_no), 'w')
    cat.create_dataset("Healpix_Index_(Position)", data=final_positions)
    cat.create_dataset("Redshift_z", data=final_redshifts)
    cat.close()

    corrected_pixels_dir = cats_dir + 'corrected_pixels/'
    if not os.path.exists(corrected_pixels_dir):
        os.makedirs(corrected_pixels_dir)

    np.savetxt(corrected_pixels_dir + 'corrected_pixels_{}.txt'.format(iter_no),
               np.transpose(corr_pixels))


if __name__ == '__main__':
    execute_poisson_sampling()
