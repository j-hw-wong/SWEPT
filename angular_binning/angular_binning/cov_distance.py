"""
Functions relating to the covariance-weighted distance statistic.
"""

import time

import gaussian_cl_likelihood.python.like_cl_gauss as like_cl  # https://github.com/robinupham/gaussian_cl_likelihood
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

import angular_binning.like_cf_gauss as like_cf


def get_dist_sq_fullskycl(cl, inv_cov):
    """
    For a set of power spectra ``cl``, get the square of the covariance-weighted distance, ``cl @ inv_cov @ cl``,
    using the fact that different l are independent to avoid the full calculation.

    Args:
        cl (2D numpy array): Power spectra with shape (n_spectra, n_ell).
        inv_cov (2D numpy array): Inverse of the covariance matrix corresponding to the flattened ``cl`` array,
                                  flattened in row-major (spectra-major) order.

    Returns:
        float: Squared covariance-weighted distance.
    """

    return np.einsum('sl,lst,tl->', cl, inv_cov, cl)


def prepare_validation(n_zbin, lmax, lmin, diag_cls_no_b_path, diag_cls_with_b_path, pos_nl_path, she_nl_path,
                       theta_min, theta_max, n_theta_bin, survey_area_sqdeg, gals_per_sqarcmin_per_zbin, sigma_e,
                       data_save_path):
    """
    Calculate the data for the covariance-weighted distance validation plots, using 3x2pt power spectra with and without
    B-modes produced with ``param_grids.load_diagonal_3x2pt_cl`` and ``load_diagonal_3x2pt_cl_with_b``, and save to file
    for fast plotting with ``plot_validation``.

    Args:
        n_zbin (int): Number of redshift bins. 1 position field and 1 shear field per redshift bin are assumed.
        lmax (int): Maximum l.
        lmin (int): Minimum l.
        diag_cls_no_b_path (str): Path to output of ``load_diagonal_3x2pt_cl``.
        diag_cls_with_b_path (str): Path to output of ``load_diagonal_3x2pt_cl_with_b``.
        pos_nl_path (str): Path to position power spectrum as a text file.
        she_nl_path (str): Path to shear power spectrum as a text file.
        theta_min (float): Minimum theta in radians.
        theta_max (float): Maximum theta in radians.
        n_theta_bin (int): Number of theta bins.
        survey_area_sqdeg (float): Survey area in square degrees.
        gals_per_sqarcmin_per_zbin (float): Average number of galaxies per square arcminute per redshift bin.
        sigma_e (float): Intrinsic galaxy ellipticity dispersion per component.
        data_save_path (str): Path to save output for fast plotting with ``plot_validation``.
    """

    # Calculate some useful quantities
    n_ell = lmax - lmin + 1
    n_field_no_b = 2 * n_zbin
    n_field_with_b = 3 * n_zbin
    n_spec_no_b = n_field_no_b * (n_field_no_b + 1) // 2
    n_spec_with_b = n_field_with_b * (n_field_with_b + 1) // 2

    # Load parameters and power spectra with and without B-modes
    with np.load(diag_cls_no_b_path) as data:
        w0 = data['w0']
        wa = data['wa']
        theory_cls_no_b = data['theory_cls'][:, :, :n_ell]
    with np.load(diag_cls_with_b_path) as data:
        w0_check = data['w0']
        wa_check = data['wa']
        theory_cls_with_b = data['theory_cls'][:, :, :n_ell]
    assert np.array_equal(w0, w0_check)
    assert np.array_equal(wa, wa_check)
    n_model = theory_cls_no_b.shape[0]
    assert theory_cls_no_b.shape == (n_model, n_spec_no_b, n_ell)
    assert theory_cls_with_b.shape == (n_model, n_spec_with_b, n_ell)

    # Add noise to the no-B Cls (with-B are for correlation function, so no noise)
    pos_nl = np.loadtxt(pos_nl_path, max_rows=n_ell)
    she_nl = np.loadtxt(she_nl_path, max_rows=n_ell)
    theory_cls_no_b[:, :n_field_no_b:2, :] += pos_nl
    theory_cls_no_b[:, 1:n_field_no_b:2, :] += she_nl

    # Identify fiducial Cls
    fid_idx = np.squeeze(np.argwhere(np.isclose(w0, -1) & np.isclose(wa, 0)))
    fid_cl_no_b = theory_cls_no_b[fid_idx, :, :]
    fid_cl_with_b = theory_cls_with_b[fid_idx, :, :]

    # Calculate distance from (-1, 0) with a direction (bottom left being negative)
    dist = np.sqrt((w0 - -1) ** 2 + (wa - 0) ** 2) * np.sign(wa)

    # Convert distance to units of sigma using the fact that on the w0-wa posterior there are 21 points inside +/- 9 sig
    # for the power spectrum, and 12 grid steps to go from -7 to +8 sig for the 10-bin correlation function
    onesig_cl = np.mean(np.diff(dist)) * (21 - 1) / 18
    onesig_cf = 12 * np.mean(np.diff(dist)) / (8 - - 7)
    dist_sigma_cl = np.abs(dist / onesig_cl)
    dist_sigma_cf = np.abs(dist / onesig_cf)

    # Calculate Cl covariance
    ell = np.arange(lmin, lmax + 1)
    cov_nl = np.zeros_like(fid_cl_no_b.T)  # because noise already added
    cl_invcovs = like_cl.cl_invcov(ell, fid_cl_no_b.T, cov_nl, n_field_no_b)

    # Calculate covariance-weighted distance
    cl_diffs = theory_cls_no_b - fid_cl_no_b[np.newaxis, ...]
    cl_cov_dist = np.sqrt([get_dist_sq_fullskycl(cl_diff, cl_invcovs) for cl_diff in cl_diffs])

    # Calculate primary transformation matrix which take you from Cls to binned CFs
    l_extrap_to = lmax  # no extrapolation needed here
    transmat, _ = like_cf.generate_cl_to_binned_cf_matrices(n_zbin, lmax, lmin, l_extrap_to, theta_min,
                                                            theta_max, n_theta_bin)

    # Calculate per-l Cl covariance matrices for fiducial Cls
    fields = ['N', 'E', 'B'] * n_zbin
    assert len(fields) == n_field_with_b
    spectra = [fields[r] + fields[r + d] for d in range(n_field_with_b) for r in range(n_field_with_b - d)]
    keep_spectra = ['B' not in spec or spec == 'BB' for spec in spectra]
    cl_covs = like_cf.calculate_cl_covs(fid_cl_with_b, n_zbin, keep_spectra, lmin)

    # Calculate CF cov with einsum
    # This requires that transmat is first reshaped so it can be indexed
    # as [cf_idx, spec_idx, ell_idx]
    n_cf = transmat.shape[0]
    n_spec_cf = cl_covs.shape[1]
    assert transmat.shape == (n_cf, n_spec_cf * n_ell)
    assert cl_covs.shape == (n_ell, n_spec_cf, n_spec_cf)
    transmat_3d = np.reshape(transmat, (n_cf, n_spec_cf, n_ell))
    cf_cov = np.einsum('ikl,jnl,lkn->ij', transmat_3d, transmat_3d, cl_covs, optimize='greedy')

    # Add noise variance
    cf_noise_var = like_cf.get_cf_noise_variance(n_zbin, theta_min, theta_max, n_theta_bin, survey_area_sqdeg,
                                                 gals_per_sqarcmin_per_zbin, sigma_e)
    cf_cov += np.diag(cf_noise_var)

    # Do some checks
    assert np.all(np.isfinite(cf_cov))
    assert np.allclose(cf_cov, cf_cov.T)
    assert np.all(np.linalg.eigvals(cf_cov) > 0)
    assert np.all(np.diag(cf_cov) > 0)

    # Invert covariance
    cf_invcov = scipy.linalg.inv(cf_cov)

    # Convert Cls to CFs for each model
    theory_cls_forcf = np.reshape(theory_cls_with_b[:, keep_spectra, :], (n_model, n_spec_cf * n_ell))
    theory_cfs = (transmat @ theory_cls_forcf.T).T

    # Loop over models, calculate (model_cf - fid_cf) @ invcov @ (model_cf - fid_cf) and square root it
    fid_cf = theory_cfs[fid_idx, :]
    cf_diffs = theory_cfs - fid_cf[np.newaxis, ...]
    cf_cov_dist = np.sqrt(np.einsum('mi,ij,mj->m', cf_diffs, cf_invcov, cf_diffs))

    # Save to disk
    header = (f'Output from {__file__}.prepare_validation input n_zbin = {n_zbin}, lmax = {lmax}, lmin = {lmin}, '
              f'diag_cls_no_b_path = {diag_cls_no_b_path}, diag_cls_with_b_path = {diag_cls_with_b_path}, '
              f'pos_nl_path = {pos_nl_path}, she_nl_path = {she_nl_path}, theta_min = {theta_min}, '
              f'theta_max = {theta_max}, n_theta_bin = {n_theta_bin}, survey_area_sqdeg = {survey_area_sqdeg}, '
              f'gals_per_sqarcmin_per_zbin = {gals_per_sqarcmin_per_zbin}, sigma_e = {sigma_e}, '
              f'at {time.strftime("%c")}')
    np.savez_compressed(data_save_path, dist_sigma_cl=dist_sigma_cl, dist_sigma_cf=dist_sigma_cf,
                        cl_cov_dist=cl_cov_dist, cf_cov_dist=cf_cov_dist, header=header)
    print('Saved ' + data_save_path)


def plot_validation(plot_data_path, plot_save_path=None):
    """
    Produce the covariance-weighted distance validation plot, using data produced with ``prepare_validation``.

    Args:
        plot_data_path (str): Path to output of ``prepare_validation``.
        plot_save_path (str, optional): Path to save plot, if required. If not supplied, plot will be displayed.
    """

    # Load data
    with np.load(plot_data_path) as data:
        dist_sigma_cl = data['dist_sigma_cl']
        dist_sigma_cf = data['dist_sigma_cf']
        cl_cov_dist = data['cl_cov_dist']
        cf_cov_dist = data['cf_cov_dist']

    # Prepare plot
    plt.rcParams.update({'font.size': 13})
    fig, ax = plt.subplots(ncols=2, figsize=(12.8, 5))
    plt.subplots_adjust(left=.06, right=.99, wspace=.1, bottom=.14, top=.98)

    # Scatter plot of the distances
    ax[0].scatter(dist_sigma_cl, cl_cov_dist)
    ax[1].scatter(dist_sigma_cf, cf_cov_dist)

    # Line for x=y
    for a in ax:
        a.axline((0, 0), (1, 1), ls='--', c='k', alpha=.2)

    # Axis labels
    big_ax = fig.add_subplot(frameon=False)
    big_ax.tick_params(labelcolor='none', bottom=False, left=False)
    big_ax.set_xlabel(r'Distance measured from posterior in $\sigma$', labelpad=15)
    ax[0].set_ylabel('Covariance-weighted distance $d$', labelpad=15)

    # Set the power spectrum x ticks manually
    ax[0].set_xticks([0, 5, 10, 15, 20])

    # Panel labels
    for a, label in zip(ax, ['Power spectrum', 'Correlation function']):
        a.annotate(label, xy=(.05, .95), xycoords='axes fraction', va='top', fontsize=14)

    if plot_save_path is not None:
        plt.savefig(plot_save_path)
        print('Saved ' + plot_save_path)
    else:
        plt.show()


def get_weighted_pbl(n_bandpowers, output_lmin, output_lmax, input_lmin=None, input_lmax=None, weighting='default'):
    """
    Get the bandpower binning matrix with different l weighting schemes.

    Options are:
        * ``default`` = l(l+1)/2π
        * ``flat`` = 1
        * ``sin`` = sin(π/l)

    Args:
        n_bandpowers (int): Number of bandpowers.
        output_lmin (int): Minimum l included in the first bandpower.
        output_lmax (int): Maxmimum l included in the last bandpower.
        input_lmin (int, optional): Minimum l included in the input (default is equal to ``output_lmin``).
        input_lmax (int, optional): Maximum l included in the input (default is equal to ``output_lmax``).
        weighting (str, optional): Weighting scheme. Options are ``default``, ``flat`` and ``sin``, as described above.
    """

    # Calculate bin boundaries (add small fraction to lmax to include it in the end bin)
    edges = np.logspace(np.log10(output_lmin), np.log10(output_lmax + 1e-5), n_bandpowers + 1)

    # Calculate input ell range and create broadcasted views for convenience
    if input_lmin is None:
        input_lmin = output_lmin
    if input_lmax is None:
        input_lmax = output_lmax
    ell = np.arange(input_lmin, input_lmax + 1)[None, ...]
    lower_edges = edges[:-1, None]
    upper_edges = edges[1:, None]

    # First calculate a boolean matrix of whether each ell is included in each bandpower,
    # then apply the l(l+1)/2π / n_l factor where n_l is the number of ells in the bin
    in_bin = (ell >= lower_edges) & (ell < upper_edges)
    n_ell = np.floor(upper_edges) - np.ceil(lower_edges) + 1
    assert np.all(n_ell > 0), 'One or more empty bandpowers. Choose fewer bandpowers.'

    # Default weighting
    if weighting == 'default':
        pbl = in_bin * ell * (ell + 1) / (2 * np.pi * n_ell)

    # Flat weighting
    elif weighting == 'flat':
        pbl = in_bin / n_ell

    # Sin weighting
    elif weighting == 'sin':
        theta = np.pi / ell
        pbl = in_bin * np.sin(theta) / n_ell

    else:
        raise ValueError(f'Unexpected weighting {weighting}')

    return pbl


def prepare_dist_vs_nbin(n_zbin, lmax, lmin, diag_cls_no_b_path, diag_cls_with_b_path, n_bps, pos_nl_path, she_nl_path,
                         theta_min, theta_max, n_theta_bins, survey_area_sqdeg, gals_per_sqarcmin_per_zbin, sigma_e,
                         data_save_path):
    """
    Calculate covariance-weighted distance vs number of angular bins for the power spectrum with three different
    weightings and the correlation function, using 3x2pt power spectra with and without B-modes produced with
    ``param_grids.load_diagonal_3x2pt_cl`` and ``load_diagonal_3x2pt_cl_with_b``, and save the result to disk for fast
    plotting with ``plot_dist_vs_nbin``.

    Args:
        n_zbin (int): Number of redshift bins. 1 position field and 1 shear field per redshift bin are assumed.
        lmax (int): Maximum l.
        lmin (int): Minimum l.
        diag_cls_no_b_path (str): Path to output of ``load_diagonal_3x2pt_cl``.
        diag_cls_with_b_path (str): Path to output of ``load_diagonal_3x2pt_cl_with_b``.
        n_bps (list): List or other sequence of numbers of bandpowers over which to iterate.
        pos_nl_path (str): Path to position power spectrum as a text file.
        she_nl_path (str): Path to shear power spectrum as a text file.
        theta_min (float): Minimum theta in radians.
        theta_max (float): Maximum theta in radians.
        n_theta_bins (list): List or other sequence of numbers of theta bins over which to iterate.
        survey_area_sqdeg (float): Survey area in square degrees.
        gals_per_sqarcmin_per_zbin (float): Average number of galaxies per square arcminute per redshift bin.
        sigma_e (float): Intrinsic galaxy ellipticity dispersion per component.
        data_save_path (str): Path to save output for fast plotting with ``plot_dist_vs_nbin``.
    """

    # Calculate useful quantities
    n_ell = lmax - lmin + 1
    n_field_no_b = 2 * n_zbin
    n_field_with_b = 3 * n_zbin
    n_spec_no_b = n_field_no_b * (n_field_no_b + 1) // 2
    n_spec_with_b = n_field_with_b * (n_field_with_b + 1) // 2
    n_n_bp = len(n_bps)
    n_n_theta_bin = len(n_theta_bins)

    # Load parameters and power spectra
    with np.load(diag_cls_no_b_path) as data:
        w0 = data['w0']
        wa = data['wa']
        theory_cls_no_b = data['theory_cls'][:, :, :n_ell]
    with np.load(diag_cls_with_b_path) as data:
        w0_check = data['w0']
        wa_check = data['wa']
        theory_cls_with_b = data['theory_cls'][:, :, :n_ell]
    n_model = len(w0)
    assert w0.shape == (n_model,)
    assert wa.shape == (n_model,)
    assert np.array_equal(w0_check, w0)
    assert np.array_equal(wa_check, wa)
    assert theory_cls_no_b.shape == (n_model, n_spec_no_b, n_ell)
    assert theory_cls_with_b.shape == (n_model, n_spec_with_b, n_ell)

    # Add noise (with-B are for correlation function, so no noise)
    pos_nl = np.loadtxt(pos_nl_path, max_rows=n_ell)
    she_nl = np.loadtxt(she_nl_path, max_rows=n_ell)
    theory_cls_no_b[:, :n_field_no_b:2, :] += pos_nl
    theory_cls_no_b[:, 1:n_field_no_b:2, :] += she_nl

    # Identify fiducial Cls
    fid_idx = np.squeeze(np.argwhere(np.isclose(w0, -1) & np.isclose(wa, 0)))
    fid_cl_no_b = theory_cls_no_b[fid_idx, :, :]
    fid_cl_with_b = theory_cls_with_b[fid_idx, :, :]

    # Calculate distance from (-1, 0) with a direction (bottom left being negative)
    dist = np.sqrt((w0 - -1) ** 2 + (wa - 0) ** 2) * np.sign(wa)

    # Convert distance to units of sigma using the fact that on the w0-wa posterior there are 21 points inside +/- 9 sig
    # for the power spectrum, and 12 grid steps to go from -7 to +8 sig for the 10-bin correlation function
    onesig_cl = np.mean(np.diff(dist)) * (21 - 1) / 18
    onesig_cf = 12 * np.mean(np.diff(dist)) / (8 - - 7)
    dist_sigma_cl = dist / onesig_cl
    dist_sigma_cf = dist / onesig_cf

    # Calculate Cl covariance
    cl_covs_no_b = np.full((n_ell, n_spec_no_b, n_spec_no_b), np.nan)
    for l in range(lmin, lmax + 1):
        print(f'Calculating Cl covariance l = {l} / {lmax}', end='\r')
        cl_covs_no_b[l - lmin, :, :] = like_cl.cl_cov(fid_cl_no_b[:, l - lmin], l, n_field_no_b)
    assert np.all(np.isfinite(cl_covs_no_b))
    print()
    assert np.all([np.all(np.linalg.eigvals(cl_cov) > 0) for cl_cov in cl_covs_no_b])

    # Arrays to hold the results, indexed as [model_idx, n_bp]
    dists_cl_default = np.full((n_model, n_n_bp), np.nan)
    dists_cl_flat = dists_cl_default.copy()
    dists_cl_sin = dists_cl_default.copy()

    # Loop over weighting schemes
    for weighting, dists in zip(['default', 'flat', 'sin'], [dists_cl_default, dists_cl_flat, dists_cl_sin]):

        # Loop over the numbers of bins
        for n_bp_idx, n_bp in enumerate(n_bps):
            print(weighting, 'n_bp =', n_bp)

            # Get bandpower binning matrix
            pbl = get_weighted_pbl(n_bp, lmin, lmax, weighting=weighting)

            # Bin all Cls (b = bandpower, l = ell, m = model, s = spectrum)
            theory_bps = np.einsum('bl,msl->msb', pbl, theory_cls_no_b)
            fid_bp = theory_bps[fid_idx, :, :]

            # Bin covariance (b = bandpower, l = ell, s = spec1, t = spec2)
            bp_covs = np.einsum('bl,lst->bst', pbl ** 2, cl_covs_no_b)
            assert np.all([np.all(np.linalg.eigvals(bp_cov) > 0) for bp_cov in bp_covs])

            # Invert covariance
            bp_invcovs = np.array([np.linalg.inv(bp_cov) for bp_cov in bp_covs])
            assert np.all([np.all(np.linalg.eigvals(bp_invcov) > 0) for bp_invcov in bp_invcovs])

            # Calculate covariance-weighted distance
            bp_diffs = theory_bps - fid_bp[np.newaxis, ...]
            dists[:, n_bp_idx] = np.sqrt([get_dist_sq_fullskycl(bp_diff, bp_invcovs) for bp_diff in bp_diffs])

    assert np.all(np.isfinite(dists_cl_default))
    assert np.all(np.isfinite(dists_cl_flat))
    assert np.all(np.isfinite(dists_cl_sin))

    # Calculate per-l Cl covariance matrices for fiducial Cls
    fields_with_b = ['N', 'E', 'B'] * n_zbin
    assert len(fields_with_b) == n_field_with_b
    spectra_with_b = [fields_with_b[row] + fields_with_b[row + diag]
                      for diag in range(n_field_with_b) for row in range(n_field_with_b - diag)]
    keep_spectra = ['B' not in spec or spec == 'BB' for spec in spectra_with_b]
    cl_covs_with_b = like_cf.calculate_cl_covs(fid_cl_with_b, n_zbin, keep_spectra, lmin)

    # Other things that are independent of the number of theta bins
    n_spec_cf = np.sum(keep_spectra)
    theory_cls_flat_cf = np.reshape(theory_cls_with_b[:, keep_spectra, :], (n_model, n_spec_cf * n_ell))

    # Loop over the numbers of theta bins
    dists_cf = np.full((n_model, n_n_theta_bin), np.nan)
    for nbin_idx, n_theta_bin in enumerate(n_theta_bins):
        print('n_theta_bin:', n_theta_bin)

        # Calculate primary transformation matrix which take you from Cls to binned CFs
        l_extrap_to = lmax  # no extrapolation needed here
        transmat, _ = like_cf.generate_cl_to_binned_cf_matrices(n_zbin, lmax, lmin, l_extrap_to, theta_min,
                                                                theta_max, n_theta_bin, verbose=False)

        # Calculate CF cov with einsum
        # This requires that transmat is first reshaped so it can be indexed
        # as [cf_idx, spec_idx, ell_idx]
        n_cf, _ = transmat.shape
        transmat_3d = np.reshape(transmat, (n_cf, n_spec_cf, n_ell))
        print('Transforming Cl covariance to CF covariance')
        cf_cov = np.einsum('ikl,jnl,lkn->ij', transmat_3d, transmat_3d, cl_covs_with_b, optimize='greedy')

        # Add noise variance
        cf_noise_var = like_cf.get_cf_noise_variance(n_zbin, theta_min, theta_max, n_theta_bin, survey_area_sqdeg,
                                                     gals_per_sqarcmin_per_zbin, sigma_e)
        cf_cov += np.diag(cf_noise_var)

        # Do some checks
        assert np.all(np.isfinite(cf_cov))
        assert np.allclose(cf_cov, cf_cov.T)
        is_pd = np.all(np.linalg.eigvals(cf_cov) > 0)
        assert is_pd
        assert np.all(np.diag(cf_cov) > 0)

        # Invert covariance
        cf_invcov = scipy.linalg.inv(cf_cov)

        # Convert Cls to CFs for each model
        theory_cfs = (transmat @ theory_cls_flat_cf.T).T

        # Loop over models, calculate (model_cf - fid_cf) @ invcov @ (model_cf - fid_cf) and square root it
        fid_cf = theory_cfs[fid_idx, :]
        cf_diffs = theory_cfs - fid_cf[np.newaxis, ...]
        dists_cf[:, nbin_idx] = np.sqrt(np.einsum('mi,ij,mj->m', cf_diffs, cf_invcov, cf_diffs))

    assert np.all(np.isfinite(dists_cf))

    # Save to disk
    header = (f'Output from {__file__}.prepare_dist_vs_nbin for input n_zbin = {n_zbin}, lmax = {lmax}, lmin = {lmin}, '
              f'diag_cls_no_b_path = {diag_cls_no_b_path}, diag_cls_with_b_path = {diag_cls_with_b_path}, '
              f'n_bps = {n_bps}, pos_nl_path = {pos_nl_path}, she_nl_path = {she_nl_path}, theta_min = {theta_min}, '
              f'theta_max = {theta_max}, n_theta_bins = {n_theta_bins}, survey_area_sqdeg = {survey_area_sqdeg}, '
              f'gals_per_sqarcmin_per_zbin = {gals_per_sqarcmin_per_zbin}, sigma_e = {sigma_e}, '
              f'at {time.strftime("%c")}')
    np.savez_compressed(data_save_path, n_bps=n_bps, n_theta_bins=n_theta_bins, dists_cl_default=dists_cl_default,
                        dists_cl_flat=dists_cl_flat, dists_cl_sin=dists_cl_sin, dists_cf=dists_cf,
                        dist_sigma_cl=dist_sigma_cl, dist_sigma_cf=dist_sigma_cf, header=header)
    print('Saved ' + data_save_path)


def plot_dist_vs_nbin(plot_data_path, plot_save_path=None):
    """
    Plot covariance-weighted distance against number of angular bins for the power spectrum with three different
    weightings and the correlation function, using data prepared with ``prepare_dist_vs_nbin``.

    Args:
        plot_data_path (str): Path to output of ``prepare_dist_vs_nbin``.
        plot_save_path (str, optional): Path to save figure, if supplied. If not supplied, figure will be displayed.
    """

    # Load data
    with np.load(plot_data_path) as data:
        n_bps = data['n_bps']
        n_theta_bins = data['n_theta_bins']
        dists_cl_default = data['dists_cl_default']
        dists_cl_flat = data['dists_cl_flat']
        dists_cl_sin = data['dists_cl_sin']
        dists_cf = data['dists_cf']
        dist_sigma_cl = data['dist_sigma_cl']

    assert np.array_equal(n_bps, n_theta_bins)  # then can just plot against n_bps

    # Prepare colour scale using power spectrum posterior distances
    max_dist_sigma = np.amax(np.abs(dist_sigma_cl))
    norm = matplotlib.colors.Normalize(-max_dist_sigma, max_dist_sigma)
    colourmap = matplotlib.cm.ScalarMappable(norm, cmap='Spectral')
    colours = [colourmap.to_rgba(ds) for ds in dist_sigma_cl]

    # Prepare plot
    plt.rcParams.update({'font.size': 13})
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12.8, 8), sharex=True)
    plt.subplots_adjust(left=.05, right=.99, bottom=.09, top=.99, wspace=.1, hspace=.05)

    # Plot the four panels: Cl default, CF, Cl flat and Cl sin
    for panel, dists in zip(np.ravel(ax), [dists_cl_default, dists_cf, dists_cl_flat, dists_cl_sin]):
        for model_dists, colour in zip(dists, colours):
            panel.plot(n_bps, model_dists, c=colour)

    # Colour bar
    cb = plt.colorbar(colourmap, ax=ax, fraction=.08, pad=.02)
    cb.set_label(r'Distance from power spectrum posterior in $\sigma$', rotation=-90, labelpad=15)

    # Shared axis labels
    big_ax = fig.add_subplot(1, 9, (1, 8), frameon=False)
    big_ax.tick_params(labelcolor='none', bottom=False, left=False)
    big_ax.set_xlabel('Number of angular bins', labelpad=15)
    big_ax.set_ylabel('Covariance-weighted distance $d$', labelpad=5)

    # Panel labels
    labels = ['Power spectrum \u2013 ' + r'$\ell \left( \ell + 1 \right)$ weighting',
              'Correlation function \u2013 ' + r'$\sin \, \theta$ weighting',
              'Power spectrum \u2013 flat weighting',
              'Power spectrum \u2013 ' + r'$\sin{\left( \pi / \ell \right)}$ weighting']
    for panel, label in zip(np.ravel(ax), labels):
        _, top = panel.get_ylim()
        panel.set_ylim(top=(1.07 * top))
        panel.annotate(label, xy=(.02, .97), xycoords='axes fraction', va='top', fontsize=14)

    if plot_save_path is not None:
        plt.savefig(plot_save_path)
        print('Saved ' + plot_save_path)
    else:
        plt.show()
