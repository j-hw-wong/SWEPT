"""
Script to compile the final cosmic shear catalogue. From the table of galaxy redshifts + position index generated in
'cat_indices.hdf5' by poisson_sample_gal_position.py, we find the shear field maps at the given redshift slice, and
assign the k, y1, y2 shear values of the pixel that each galaxy is found in. When compiling the final catalogue table,
we then introduce Gaussian and catastrophic photo-z errors into the Redshift values, adding an extra column for each
phenomenon. The 'True_Redshift' is the underlying p(z)' the 'Gaussian_Redshift' is the observed redshift with Gaussian
photo-z errors only, and the 'Redshift_z' is the final observed redshift including catastrophic photo-z injection.
The final catalogue is saved as 'master_cat_poisson_sampled.hdf5' on disk - repeated over a given number of
realisations/iterations.
"""


import os
import h5py
import configparser
import numpy as np
from random import choices


def compile_cat_config(pipeline_variables_path):

    """
    Set up a config dictionary of pipeline parameters and input quantites describing the photo-z error distributions

    Parameters
    ----------
    pipeline_variables_path (str):  Path to pipeline variables .ini file

    Returns
    -------
    Dictionary of pipeline parameters
    """

    config = configparser.ConfigParser()
    config.read(pipeline_variables_path)

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

    save_dir = str(config['simulation_setup']['SIMULATION_SAVE_DIR'])
    iter_no = int(float(os.environ['ITER_NO']))

    shape_noise_mean = 0
    shape_noise_sigma = float(config['compile_cat']['SIGMA_SHEAR'])

    photo_z_noise_mean = 0
    photo_z_noise_sigma = float(config['compile_cat']['SIGMA_PHOT'])

    cat_photo_z_frac = float(config['compile_cat']['CAT_PHOTO_Z_FRAC'])
    cat_photo_z_sigma = float(config['compile_cat']['CAT_PHOTO_Z_SIGMA'])

    # Prepare config dictionary
    config_dict = {
        'save_dir': save_dir,
        'zmin': zmin,
        'zmax': zmax,
        'dz': dz,
        'z_sample': z_sample,
        'iter_no': iter_no,
        'shape_noise_mean': shape_noise_mean,
        'shape_noise_sigma': shape_noise_sigma,
        'photo_z_noise_mean': photo_z_noise_mean,
        'photo_z_noise_sigma': photo_z_noise_sigma,
        'cat_photo_z_frac': cat_photo_z_frac,
        'cat_photo_z_sigma': cat_photo_z_sigma
    }

    return config_dict


def generate_cat_err_sig(redshifts, lambda_1, lambda_2, sig):

    """
    Function to inject catastrophic photo-z errors into the redshift sample based on a confusion of two given
    wavelength lines and error distribution

    Parameters
    ----------
    redshifts (arr):    Array of galaxy redshifts to inject catastrophic photo-zs into
    lambda_1 (float):   Wavelength of first given spectral line
    lambda_2 (float):   Wavelength of second given spectral line
    sig (float):        Sigma spread describing the error distribution around where the pair confusion line is found

    Returns
    -------
    Array of galaxy redshifts with catastrophic photo-z errors
    """

    cat_z_mus = ((1+redshifts)*(lambda_1/lambda_2))-1
    return np.random.normal(cat_z_mus, sig*np.ones(len(cat_z_mus)), size=len(cat_z_mus))


def split_z_chunks(a, n):

    """
    Convenience function to split a chunk of galaxy redshifts into equal sub-chunks (used to distribute all pairs of
    photo-z confusion between)

    Parameters
    ----------
    a (arr):    Array of redshift values
    n (int):    Number of chunks to split data into

    Returns
    -------
    Array of n sub-samples that the original data array a has been split into
    """

    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]


def execute_cat_compilation():

    """
    Execute the compilation of the final master catalogue. First load in galaxy pixel indices, assign the shear k, y1,
    y2 values based on the shear field maps at the same redshift slice, then inject both Gaussian and catastrophic
    photo-z errors.
    """

    pipeline_variables_path = os.environ['PIPELINE_VARIABLES_PATH']
    compile_cat_config_dict = compile_cat_config(pipeline_variables_path=pipeline_variables_path)

    save_dir = compile_cat_config_dict['save_dir']
    iter_no = compile_cat_config_dict['iter_no']
    z_sample = compile_cat_config_dict['z_sample']

    shape_noise_mean = compile_cat_config_dict['shape_noise_mean']
    shape_noise_sigma = compile_cat_config_dict['shape_noise_sigma']

    photo_z_noise_mean = compile_cat_config_dict['photo_z_noise_mean']
    photo_z_noise_sigma = compile_cat_config_dict['photo_z_noise_sigma']

    cat_photo_z_frac = compile_cat_config_dict['cat_photo_z_frac']
    cat_photo_z_sigma = compile_cat_config_dict['cat_photo_z_sigma']

    dat = h5py.File(save_dir + 'cat_products/cat_indices/cat_indices_iter_{}.hdf5'.format(iter_no), 'r')
    inds = np.array(dat.get('Healpix_Index_(Position)'))
    zs = np.array(dat.get('Redshift_z'))

    dat.close()

    shear1_vals = []
    shear2_vals = []

    shear1_vals_true = []
    shear2_vals_true = []

    convergence_vals = []

    ras = []
    decs = []

    phot_zs = []

    for k in range(len(z_sample)):
        id_arr = np.where(np.asarray(zs) == z_sample[k])
        zs_sub = zs[id_arr]
        zs_sub = zs_sub.round(decimals=2)
        inds_sub = inds[id_arr]

        shape_noise_elements = np.random.normal(shape_noise_mean, shape_noise_sigma, len(inds_sub))

        photo_z_noise_sigmas = (1 + zs_sub) * photo_z_noise_sigma
        photo_z_noise_means = np.zeros(len(zs_sub)) + photo_z_noise_mean

        photo_z_noise = np.random.normal(photo_z_noise_means, photo_z_noise_sigmas, len(photo_z_noise_sigmas))
        obs_zs = zs_sub + photo_z_noise
        phot_zs = np.concatenate((phot_zs, obs_zs))

        conv_fname = save_dir + 'flask/interp_maps/convergence/iter_{}/map_zslice_{}.hdf5'.format(iter_no, (k + 1))
        shear_fname = save_dir + 'flask/interp_maps/shear/iter_{}/map_zslice_{}.hdf5'.format(iter_no, (k + 1))

        with h5py.File(conv_fname, 'r') as f:
            map_ra = np.array(f.get('RA'))
            map_dec = np.array(f.get('Dec'))
            map_conv = np.array(f.get('Convergence'))
            ra = map_ra[inds_sub]
            dec = map_dec[inds_sub]
            conv = map_conv[inds_sub]
            ras = np.concatenate((ras, ra))
            decs = np.concatenate((decs, dec))
            convergence_vals = np.concatenate((convergence_vals, conv))

        with h5py.File(shear_fname, 'r') as f:
            map_y1 = np.array(f.get('Shear_y1'))
            map_y2 = np.array(f.get('Shear_y2'))

            y1 = map_y1[inds_sub]
            y2 = map_y2[inds_sub]

            shear1_vals_true = np.concatenate((shear1_vals_true, np.asarray(y1)))
            shear2_vals_true = np.concatenate((shear2_vals_true, np.asarray(y2)))

            y1 = np.asarray(y1) + np.asarray(shape_noise_elements) / np.sqrt(2)
            y2 = np.asarray(y2) + np.asarray(shape_noise_elements) / np.sqrt(2)

            shear1_vals = np.concatenate((shear1_vals, y1))
            shear2_vals = np.concatenate((shear2_vals, y2))

    inds = np.asarray(inds)
    inds = inds.astype(int)

    cat_sav_dir = save_dir + 'cat_products/master_cats/'
    if not os.path.exists(cat_sav_dir):
        os.makedirs(cat_sav_dir)

    ras = np.float32(ras)
    decs = np.float32(decs)
    zs = np.float32(zs)
    phot_zs = np.float32(phot_zs)
    final_phot_zs = np.copy(phot_zs)  # will inject catastrophic errors into this column
    final_phot_zs = np.float32(final_phot_zs)
    convergence_vals = np.float32(convergence_vals)
    shear1_vals = np.float32(shear1_vals)
    shear2_vals = np.float32(shear2_vals)
    inds = np.float32(inds)

    if cat_photo_z_frac != 0:
        # Let's inject some catastrophic photo-zs

        # Angstroms
        Ly_alpha = 1216
        Ly_break = 912
        Balmer = 3700
        D4000 = 4000

        break_rfs = [Ly_alpha, Ly_alpha]  # , Ly_break, Ly_break] - could include more pair confusions
        break_catas = [Balmer, D4000]  # , Balmer, D4000] - could include more pair confusions

        rand_zs_ids = choices(range(len(zs)), k=int(len(zs) * cat_photo_z_frac))
        rand_zs_ids = np.asarray(rand_zs_ids)
        rand_zs_ids_chunks = split_z_chunks(rand_zs_ids, 4)

        final_phot_zs[rand_zs_ids_chunks[0]] = generate_cat_err_sig(zs[rand_zs_ids_chunks[0]], break_rfs[0],
                                                                    break_catas[0], cat_photo_z_sigma)
        final_phot_zs[rand_zs_ids_chunks[1]] = generate_cat_err_sig(zs[rand_zs_ids_chunks[1]], break_rfs[1],
                                                                    break_catas[1], cat_photo_z_sigma)
        final_phot_zs[rand_zs_ids_chunks[2]] = generate_cat_err_sig(zs[rand_zs_ids_chunks[2]], break_catas[0],
                                                                    break_rfs[0], cat_photo_z_sigma)
        final_phot_zs[rand_zs_ids_chunks[3]] = generate_cat_err_sig(zs[rand_zs_ids_chunks[3]], break_catas[1],
                                                                    break_rfs[1], cat_photo_z_sigma)

    master_cat = h5py.File(cat_sav_dir + 'master_cat_poisson_sampled_%s.hdf5' % iter_no, 'w')
    master_cat.create_dataset("RA", data=ras, compression="gzip")
    master_cat.create_dataset("Dec", data=decs, compression="gzip")
    master_cat.create_dataset("Redshift_z", data=final_phot_zs, compression="gzip")
    master_cat.create_dataset("Gaussian_Redshift_z", data=phot_zs, compression="gzip")
    master_cat.create_dataset("True_Redshift_z", data=zs, compression="gzip")
    master_cat.create_dataset("Convergence", data=convergence_vals, compression="gzip")
    master_cat.create_dataset("Shear_y1", data=shear1_vals, compression="gzip")
    master_cat.create_dataset("Shear_y2", data=shear2_vals, compression="gzip")
    master_cat.create_dataset("Shear_y1_true", data=shear1_vals_true, compression="gzip")
    master_cat.create_dataset("Shear_y2_true", data=shear2_vals_true, compression="gzip")
    master_cat.create_dataset("Healpix_Index_(Position)", data=inds, compression="gzip")
    master_cat.close()


if __name__ == '__main__':
    execute_cat_compilation()
