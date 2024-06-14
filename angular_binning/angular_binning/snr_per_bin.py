"""
Functions for plotting the signal to noise per angular bin.
"""

import math
import os.path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import angular_binning.like_cf_gauss as like_cf


DEG_TO_RAD = math.pi / 180.0


def plot_cl_cf(diag_she_cl_path, she_nl_path, lmin, lmax, theta_min, theta_max, n_theta_bin, survey_area_sqdeg,
               gals_per_sqarcmin, sigma_e, l_extrap_to=60000, plot_save_dir=None):
    """
    Produce plots of signal-to-noise per element for both the unbinned power spectrum and the binned correlation
    function, using data produced with ``param_grids.load_diagonal_shear_cl``.

    Args:
        diag_she_cl_path (str): Path to output of ``param_grids.load_diagonal_shear_cl``.
        she_nl_path (str): Path to shear noise power spectrum as a text file.
        lmin (int): Minimum l.
        lmax (int): Maximum l.
        theta_min (float): Minimum theta.
        theta_max (float): Maximum theta.
        n_theta_bin (int): Number of theta bins.
        survey_area_sqdeg (float): Survey area in square degrees, used to calculate the noise variance for the
                                   correlation function.
        gals_per_sqarcmin (float): Average number of galaxies per square arcminute per redshift bin, used to calculate
                                   the noise variance for the correlation function.
        sigma_e (float): Intrinsic ellipticity dispersion per component, used to calculate the noise variance for the
                         correlation function.
        l_extrap_to (int, optional): The power spectrum is extrapolated to this l prior to the Cl-to-CF transform for
                                     stability, using a l(l+1)-weighted linear extrapolation. Default 60000.
        plot_save_dir (str, optional): Directory to save the two plots into, if supplied. If not supplied, plots are
                                       displayed.
    """

    # Load parameters and power spectra
    with np.load(diag_she_cl_path) as data:
        w0 = data['w0']
        wa = data['wa']
        cls_nonoise = data['shear_cl_bin_1_1']

    # Add noise
    n_ell = lmax - lmin + 1
    nl = np.loadtxt(she_nl_path, max_rows=n_ell)
    cls_ = cls_nonoise + nl

    # Do some consistency checks
    n_samp = len(w0)
    assert w0.shape == (n_samp,)
    assert wa.shape == (n_samp,)
    assert cls_.shape == (n_samp, n_ell)

    # Identify fiducial Cls
    fid_idx = np.squeeze(np.argwhere(np.isclose(w0, -1) & np.isclose(wa, 0)))
    fid_cl = cls_[fid_idx, :]
    ell = np.arange(lmin, lmax + 1)
    fid_cl_err = np.sqrt(2 * fid_cl ** 2 / (2 * ell + 1))

    # Calculate distance from (-1, 0) with a direction (bottom left being negative)
    dist = np.sqrt((w0 - -1) ** 2 + (wa - 0) ** 2) * np.sign(wa)

    # Convert distance to units of sigma using the fact that we have 21 points inside +/- 9 sig
    # (on the w0-wa posterior from lmax 2000 power spectrum)
    onesig = np.mean(np.diff(dist)) * (21 - 1) / 18
    dist_sigma = dist / onesig

    # Use a diverging colour map over this range
    max_dist_sigma = np.amax(np.abs(dist_sigma))
    norm = matplotlib.colors.Normalize(-max_dist_sigma, max_dist_sigma)
    colour = matplotlib.cm.ScalarMappable(norm, cmap='Spectral')

    # Prepare plot
    plt.rcParams.update({'font.size': 13})
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(12.8, 7.9), gridspec_kw={'height_ratios': (2, 1)})
    plt.subplots_adjust(left=.09, right=.99, bottom=.07, top=.97, hspace=0)

    # Plot all power spectra and the difference from the fiducial model
    cl_fac = ell * (ell + 1) / (2 * np.pi)
    for cl, dist_sig in zip(cls_, dist_sigma):
        ax[0].plot(ell, cl_fac * cl, alpha=.5, color=colour.to_rgba(dist_sig))
        ax[1].plot(ell, (cl - fid_cl) / fid_cl_err, alpha=.5, color=colour.to_rgba(dist_sig))

    # Add a few cosmic variance error bars
    err_ell = np.array([500, 1000, 1500, 2000])
    err_ell_idx = err_ell - lmin
    ax[0].errorbar(err_ell, cl_fac[err_ell_idx] * fid_cl[err_ell_idx],
                   yerr=(cl_fac[err_ell_idx] * 0.5 * fid_cl_err[err_ell_idx]), lw=2, c='black', zorder=5, capsize=5,
                   ls='None', label=r'Cosmic variance + noise $\sqrt{Var (C_\ell)}$')

    # Labels, legend and colour bar
    ax[1].set_xlabel(r'$\ell$')
    ax[0].set_ylabel(r'$C_\ell \times \ell (\ell + 1) ~ / ~ 2 \pi$')
    ax[1].set_ylabel(r'$(C_\ell - C_\ell^\mathrm{fid}) ~ / ~ \sqrt{\mathrm{Var}(C_\ell)}$')
    ax[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    fig.align_ylabels()
    ax[0].legend(frameon=False, title='Bin 1 shear')
    cb = plt.colorbar(colour, ax=ax, fraction=.10, pad=.01)
    cb.set_label(r'Posterior distance from fiducial model in $\sigma$' + '\n', rotation=-90,
                 labelpad=25)

    if plot_save_dir is not None:
        plot_save_path = os.path.join(plot_save_dir, 'cl_perl.pdf')
        plt.savefig(plot_save_path)
        print('Saved ' + plot_save_path)
    else:
        plt.show()

    # Calculate theta range
    theta_bin_edges = np.logspace(np.log10(theta_min), np.log10(theta_max), n_theta_bin + 1)

    # Generate Cl -> binned CF matrix (for xi_plus)
    _, cl2cf_22plus, _ = like_cf.get_cl2cf_matrices(theta_bin_edges, lmin, l_extrap_to)

    # Extrapolate fiducial power spectrum up to l_extrap_to and zero it below lmax
    fid_cl = cls_nonoise[fid_idx, :]
    extrap_mat = get_extrap_mat(lmin, lmax, l_extrap_to)
    fid_cl_extrap = extrap_mat @ fid_cl

    # Transform it with transmat to obtain stabilisation vector
    stabl_vec = cl2cf_22plus @ fid_cl_extrap

    # Now trim transmat to lmax
    cl2cf_22plus = cl2cf_22plus[:, :(lmax - lmin + 1)]

    # Obtain fiducial CF
    fid_cf = cl2cf_22plus @ fid_cl + stabl_vec

    # Calculate error on fiducial CF, including noise
    fid_cl_var = 2 * fid_cl ** 2 / (2 * ell + 1)
    fid_cf_cov_nonoise = np.einsum('il,jl,l->ij', cl2cf_22plus, cl2cf_22plus, fid_cl_var)

    # Noise contribution
    survey_area_sterad = survey_area_sqdeg * (DEG_TO_RAD ** 2)
    gals_per_sterad = gals_per_sqarcmin * (60 / DEG_TO_RAD) ** 2
    cos_theta = np.cos(theta_bin_edges)
    bin_area_new = 2 * np.pi * -1 * np.diff(cos_theta)
    npairs = 0.5 * survey_area_sterad * bin_area_new * (gals_per_sterad ** 2) # Friedrich et al. eq 65
    fid_cf_noise_var = 2 * sigma_e ** 4 / npairs
    fid_cf_err = np.sqrt(np.diag(fid_cf_cov_nonoise) + fid_cf_noise_var)

    # Apply trimmed transmat to each power spectrum and add stabilisation vector, and plot
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(12.8, 7.9), gridspec_kw={'height_ratios': (2, 1)})
    plt.subplots_adjust(left=.09, right=.99, bottom=.07, top=.97, hspace=0)
    bin_edges_deg = np.degrees(theta_bin_edges)
    bin_centres_deg = bin_edges_deg[:-1] + 0.5 * np.diff(bin_edges_deg)
    for cl, dist_sig in zip(cls_nonoise, dist_sigma):
        cf = cl2cf_22plus @ cl + stabl_vec
        cf_diff = (cf - fid_cf) / fid_cf_err
        line_args = {'alpha': .5, 'color': colour.to_rgba(dist_sig)}
        ax[0].step(bin_edges_deg, np.pad(cf, (0, 1), mode='edge'), where='post', **line_args)
        ax[1].step(bin_edges_deg, np.pad(cf_diff, (0, 1), mode='edge'), where='post', **line_args)

    # Add error bars
    bin_centres_deg = bin_edges_deg[:-1] + 0.5 * np.diff(bin_edges_deg)
    ax[0].errorbar(bin_centres_deg, fid_cf, yerr=(0.5 * fid_cf_err), lw=2, c='black', zorder=5, capsize=5,
                    ls='None', label=r'Cosmic variance + noise $\sqrt{Var (\xi+)}$')

    # Labels, legend and colour bar
    plt.xscale('log')
    ax[1].set_xlabel(r'$\theta$ (deg)')
    ax[0].set_ylabel(r'$\xi^+ (\theta)$')
    ax[1].set_ylabel(r'$(\xi^+ - \xi^+_\mathrm{fid}) ~ / ~ \sqrt{\mathrm{Var}(\xi^+)}$')
    fig.align_ylabels()
    ax[0].legend(frameon=False, title='Bin 1 shear')
    cb = plt.colorbar(colour, ax=ax, fraction=.10, pad=.01)
    cb.set_label(r'Posterior distance from fiducial model in $\sigma$' + '\n(from power spectrum)', rotation=-90,
                 labelpad=25)

    if plot_save_dir is not None:
        plot_save_path = os.path.join(plot_save_dir, 'cf_perbin.pdf')
        plt.savefig(plot_save_path)
        print('Saved ' + plot_save_path)
    else:
        plt.show()


def plot_cf_nbin(diag_she_cl_path, lmin, lmax, theta_min, theta_max, n_bin_1, n_bin_2, survey_area_sqdeg,
                 gals_per_sqarcmin, sigma_e, l_extrap_to=60000, plot_save_path=None):
    """
    Plots signal-to-noise per bin for the full-sky correlation function for two numbers of bins side-by-side, using data
    produced with ``param_grids.load_diagonal_shear_cl``.

    Args:
        diag_she_cl_path (str): Path to output of ``param_grids.load_diagonal_shear_cl``.
        lmin (int): Minimum l.
        lmax (int): Maximum l.
        theta_min (float): Minimum theta.
        theta_max (float): Maximum theta.
        n_bin_1 (int): Number of theta bins in the left panel.
        n_bin_2 (int): Number of theta bins in the right panel.
        survey_area_sqdeg (float): Survey area in square degrees.
        gals_per_sqarcmin (float): Average number of galaxies per square arcminute per redshift bin.
        sigma_e (float): Intrinsic ellipticity dispersion per component.
        l_extrap_to (int, optional): The power spectrum is extrapolated to this l prior to the Cl-to-CF transform for
                                     stability, using a l(l+1)-weighted linear extrapolation. Default 60000.
        plot_save_path (str, optional): Path to save the plot, if supplied. If not supplied, plot is displayed.
    """

    # Load parameters and power spectra
    with np.load(diag_she_cl_path) as data:
        w0 = data['w0']
        wa = data['wa']
        cls_nonoise = data['shear_cl_bin_1_1']

    # Do some consistency checks
    n_samp = len(w0)
    assert w0.shape == (n_samp,)
    assert wa.shape == (n_samp,)

    # Identify fiducial Cls
    fid_idx = np.squeeze(np.argwhere(np.isclose(w0, -1) & np.isclose(wa, 0)))
    ell = np.arange(lmin, lmax + 1)

    # Calculate distance from (-1, 0) with a direction (bottom left being negative)
    dist = np.sqrt((w0 - -1) ** 2 + (wa - 0) ** 2) * np.sign(wa)

    # Convert distance to units of sigma using the fact that we have 21 points inside +/- 9 sig
    # (on the w0-wa posterior from lmax 2000 power spectrum)
    onesig = np.mean(np.diff(dist)) * (21 - 1) / 18
    dist_sigma = dist / onesig

    # Use a diverging colour map over this range
    max_dist_sigma = np.amax(np.abs(dist_sigma))
    norm = matplotlib.colors.Normalize(-max_dist_sigma, max_dist_sigma)
    colour = matplotlib.cm.ScalarMappable(norm, cmap='Spectral')

    # Calculate theta range
    theta_bin_edges_1 = np.logspace(np.log10(theta_min), np.log10(theta_max), n_bin_1 + 1)
    theta_bin_edges_2 = np.logspace(np.log10(theta_min), np.log10(theta_max), n_bin_2 + 1)

    # Generate Cl -> binned CF matrix (for xi_plus)
    _, cl2cf_22plus_1, _ =  like_cf.get_cl2cf_matrices(theta_bin_edges_1, lmin, l_extrap_to)
    _, cl2cf_22plus_2, _ =  like_cf.get_cl2cf_matrices(theta_bin_edges_2, lmin, l_extrap_to)

    # Extrapolate fiducial power spectrum up to l_extrap_to and zero it below lmax
    fid_cl = cls_nonoise[fid_idx, :]
    extrap_mat = get_extrap_mat(lmin, lmax, l_extrap_to)
    fid_cl_extrap = extrap_mat @ fid_cl

    # Transform it with transmat to obtain stabilisation vector
    stabl_vec_1 = cl2cf_22plus_1 @ fid_cl_extrap
    stabl_vec_2 = cl2cf_22plus_2 @ fid_cl_extrap

    # Now trim transmat to lmax
    cl2cf_22plus_1 = cl2cf_22plus_1[:, :(lmax - lmin + 1)]
    cl2cf_22plus_2 = cl2cf_22plus_2[:, :(lmax - lmin + 1)]

    # Obtain fiducial CF
    fid_cf_1 = cl2cf_22plus_1 @ fid_cl + stabl_vec_1
    fid_cf_2 = cl2cf_22plus_2 @ fid_cl + stabl_vec_2

    # Calculate error on fiducial CF, including noise
    fid_cl_var = 2 * fid_cl ** 2 / (2 * ell + 1)
    fid_cf_cov_nonoise_1 = np.einsum('il,jl,l->ij', cl2cf_22plus_1, cl2cf_22plus_1, fid_cl_var)
    fid_cf_cov_nonoise_2 = np.einsum('il,jl,l->ij', cl2cf_22plus_2, cl2cf_22plus_2, fid_cl_var)

    # Noise contribution
    survey_area_sterad = survey_area_sqdeg * (DEG_TO_RAD ** 2)
    gals_per_sterad = gals_per_sqarcmin * (60 / DEG_TO_RAD) ** 2
    cos_theta_1 = np.cos(theta_bin_edges_1)
    cos_theta_2 = np.cos(theta_bin_edges_2)
    bin_area_1 = 2 * np.pi * -1 * np.diff(cos_theta_1)
    bin_area_2 = 2 * np.pi * -1 * np.diff(cos_theta_2)
    npairs_1 = 0.5 * survey_area_sterad * bin_area_1 * (gals_per_sterad ** 2) # Friedrich et al. eq 65
    npairs_2 = 0.5 * survey_area_sterad * bin_area_2 * (gals_per_sterad ** 2)
    fid_cf_noise_var_1 = 2 * sigma_e ** 4 / npairs_1
    fid_cf_noise_var_2 = 2 * sigma_e ** 4 / npairs_2
    fid_cf_err_1 = np.sqrt(np.diag(fid_cf_cov_nonoise_1) + fid_cf_noise_var_1)
    fid_cf_err_2 = np.sqrt(np.diag(fid_cf_cov_nonoise_2) + fid_cf_noise_var_2)

    # Prepare plot
    plt.rcParams.update({'font.size': 13})
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(12.8, 7.9), gridspec_kw={'height_ratios': (2, 1)})
    plt.subplots_adjust(left=.07, right=1, bottom=.07, top=.97, hspace=0, wspace=.12)

    # Apply trimmed transmat to each power spectrum and add stabilisation vector, and plot
    bin_edges_deg_1 = np.degrees(theta_bin_edges_1)
    bin_edges_deg_2 = np.degrees(theta_bin_edges_2)
    for cl, dist_sig in zip(cls_nonoise, dist_sigma):
        cf_1 = cl2cf_22plus_1 @ cl + stabl_vec_1
        cf_2 = cl2cf_22plus_2 @ cl + stabl_vec_2
        cf_diff_1 = (cf_1 - fid_cf_1) / fid_cf_err_1
        cf_diff_2 = (cf_2 - fid_cf_2) / fid_cf_err_2
        step_args = {'where': 'post', 'alpha': .5, 'color': colour.to_rgba(dist_sig)}
        ax[0, 0].step(bin_edges_deg_1, np.pad(cf_1, (0, 1), mode='edge'), **step_args)
        ax[0, 1].step(bin_edges_deg_2, np.pad(cf_2, (0, 1), mode='edge'), **step_args)
        ax[1, 0].step(bin_edges_deg_1, np.pad(cf_diff_1, (0, 1), mode='edge'), **step_args)
        ax[1, 1].step(bin_edges_deg_2, np.pad(cf_diff_2, (0, 1), mode='edge'), **step_args)

    # Add error bars
    log_bin_edges_deg_1 = np.log(bin_edges_deg_1)
    log_bin_edges_deg_2 = np.log(bin_edges_deg_2)
    bin_log_centres_deg_1 = np.exp(log_bin_edges_deg_1[:-1] + 0.5 * np.diff(log_bin_edges_deg_1))
    bin_log_centres_deg_2 = np.exp(log_bin_edges_deg_2[:-1] + 0.5 * np.diff(log_bin_edges_deg_2))
    error_args = {'lw': 2, 'c': 'black', 'zorder': 5, 'capsize': 5, 'ls': 'None',
                  'label': r'Cosmic variance + noise $\sqrt{Var (\xi+)}$'}
    ax[0, 0].errorbar(bin_log_centres_deg_1, fid_cf_1, yerr=(0.5 * fid_cf_err_1), **error_args)
    ax[0, 1].errorbar(bin_log_centres_deg_2, fid_cf_2, yerr=(0.5 * fid_cf_err_2), **error_args)

    # Log scale and axis labels
    plt.xscale('log')
    ax[1, 0].set_xlabel(r'$\theta$ (deg)')
    ax[1, 1].set_xlabel(r'$\theta$ (deg)')
    ax[0, 0].set_ylabel(r'$\xi^+ (\theta)$')
    ax[1, 0].set_ylabel(r'$(\xi^+ - \xi^+_\mathrm{fid}) ~ / ~ \sqrt{\mathrm{Var}(\xi^+)}$')
    fig.align_ylabels()

    # Panel labels
    annot_args = {'xy': (.95, .95), 'xycoords': 'axes fraction', 'ha': 'right', 'va': 'top', 'fontsize': 14}
    ax[0, 0].annotate(f'{n_bin_1} $\\theta$ bin{"s" if n_bin_1 > 1 else ""}', **annot_args)
    ax[0, 1].annotate(f'{n_bin_2} $\\theta$ bin{"s" if n_bin_2 > 1 else ""}', **annot_args)

    # Colour bar
    cb = plt.colorbar(colour, ax=ax, fraction=.10, pad=.01)
    cb.set_label(r'Posterior distance from fiducial model in $\sigma$' + '\n(from power spectrum)', rotation=-90,
                 labelpad=25)

    if plot_save_path is not None:
        plt.savefig(plot_save_path)
        print('Saved ' + plot_save_path)
    else:
        plt.show()


def get_extrap_mat(lmin, lmax_in, l_extrap_to):
    """
    Generate the power spectrum extrapolation matrix, which is used to extrapolate the power spectrum to high l
    to stabilise the Cl-to-CF transform.

    This matrix should be (pre-)multiplied by the fiducial power spectrum, then all (pre-)multiplied by the Cl-to-CF
    transformation matrix, to produce a 'stabilisation vector' which can be added to any correlation function vector to
    stabilise it. Generally the same stabilisation vector should be used for all points in parameter space, to avoid
    biases. Note that the extrapolation matrix zeros all power below lmax_in, i.e. it does not give a concatenation of
    the original power spectrum and the extrapolated section, but just solely the extrapolated section.

    The extrapolation is linear with an l(l+1) weighting, achieved using a block matrix. See extrapolation_equations.pdf
    for the derivation of its elements.

    Args:
        lmin (int): Minimum l in the power spectrum.
        lmax_in (int): Maximum l prior to extrapolation.
        l_extrap_to (int): Maximum l to which to extrapolate.

    Returns:
        2D numpy array: Extrapolation matrix.
    """

    zero_top = np.zeros((lmax_in - lmin + 1, lmax_in - lmin + 1))
    zero_bottom = np.zeros((l_extrap_to - lmax_in, lmax_in - lmin + 1 - 2))
    ell_extrap = np.arange(lmax_in + 1, l_extrap_to + 1)
    penul_col = (-ell_extrap + lmax_in) * lmax_in * (lmax_in - 1) / (ell_extrap * (ell_extrap + 1))
    final_col = (ell_extrap - lmax_in + 1) * lmax_in * (lmax_in + 1) / (ell_extrap * (ell_extrap + 1))
    extrap_mat = np.block([[zero_top], [zero_bottom, penul_col[:, np.newaxis], final_col[:, np.newaxis]]])

    return extrap_mat
