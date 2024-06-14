"""
Functions to plot statistical error as a function of number of angular bins.
"""

import time

import gaussian_cl_likelihood.python.posteriors # https://github.com/robinupham/gaussian_cl_likelihood
import matplotlib.gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.special


def area_vs_nbin(cl_like_filemask, cf_like_filemask, contour_levels_sig, n_bps, n_theta_bins, save_path=None):
    """
    Plot average area of w0-wa contours against number of angular bins for the power spectrum and correlation function
    side-by-side.

    Args:
        cl_like_filemask (str): Path to power spectrum likelihood files as output by
                                ``loop_likelihood_nbin.like_bp_gauss_loop_nbin`` (full sky) or
                                ``like_bp_gauss_mix_loop_nbin`` (cut sky), with ``{n_bp}`` placeholder.
        cf_like_filemask (str): Path to correlation function likelihood files as output by
                                ``loop_likelihood_nbin.like_cf_gauss_loop_nbin``, with ``{n_bin}`` placeholder.
        contour_levels_sig (list): List or other sequence of sigma confidence levels over which to average the contour
                                   area; can be integers e.g. ``[1, 2, 3]`` or not e.g. ``np.arange(1, 3, 100)``.
        n_bps (list): list or other sequence of numbers of bandpowers over which to iterate.
        n_theta_bins (list): List or other sequence of numbers of theta bins over which to iterate.
        save_path (str, optional): Path to save the combined plot, if supplied. If not supplied, plot will be shown.
    """

    # Calculate contour levels in probability
    contour_levels = [0.] + [scipy.special.erf(contour_level / np.sqrt(2)) for contour_level in contour_levels_sig]
    print(contour_levels)
    # Arrays to hold results: first axis is contour level, second is number of bins
    n_contour_levels = len(contour_levels_sig)
    print(n_contour_levels)
    n_n_bp = len(n_bps)
    n_n_tbin = len(n_theta_bins)
    cl_areas = np.full((n_contour_levels, n_n_bp), np.nan)
    print(cl_areas)
    cf_areas = np.full((n_contour_levels, n_n_tbin), np.nan)

    # Power spectrum: loop over numbers of bandpowers
    for n_bp_idx, n_bp in enumerate(n_bps):
        print(f'Power spectrum: n_bp = {n_bp}', end='\r')

        # Load log-likelihood
        log_like_path = cl_like_filemask.format(n_bp=n_bp)
        x_vals, y_vals, log_like = np.loadtxt(log_like_path, unpack=True)

        # Convert log-likelihood to unnormalised posterior (flat prior) while aiming to prevent over/underflows
        log_like = log_like - np.amax(log_like) - 0.5 * np.amin(log_like - np.amax(log_like))
        post = np.exp(log_like)

        # Form x and y grids and determine grid cell size (requires and checks for regular grid)
        x_vals_unique = np.unique(x_vals)
        dx = x_vals_unique[1] - x_vals_unique[0]
        assert np.allclose(np.diff(x_vals_unique), dx)
        y_vals_unique = np.unique(y_vals)
        dy = y_vals_unique[1] - y_vals_unique[0]
        dxdy = dx * dy
        assert np.allclose(np.diff(y_vals_unique), dy)
        x_grid, y_grid = np.meshgrid(x_vals_unique, y_vals_unique)

        # Grid posterior and convert to confidence intervals
        post_grid = scipy.interpolate.griddata((x_vals, y_vals), post, (x_grid, y_grid), fill_value=0)

        # Convert to confidence
        conf_grid = gaussian_cl_likelihood.python.posteriors.posterior_grid_to_confidence_levels(post_grid, dxdy)
        print(conf_grid)
        # Measure area
        cl_areas[:, n_bp_idx] = np.count_nonzero((conf_grid[..., np.newaxis] < contour_levels[1:]), axis=(0, 1)) * dxdy

    print('Power spectrum: done          ')

    # Correlation function - loop over numbers of theta bins
    for nbin_idx, nbin in enumerate(n_theta_bins):
        print(f'Correlation function: nbin = {nbin}', end='\r')

        # Load log-likelihood
        log_like_path = cf_like_filemask.format(n_bin=nbin)
        x_vals, y_vals, log_like = np.loadtxt(log_like_path, unpack=True)

        # Convert log-likelihood to unnormalised posterior (flat prior) while aiming to prevent over/underflows
        log_like = log_like - np.amax(log_like) - 0.5 * np.amin(log_like - np.amax(log_like))
        post = np.exp(log_like)

        # Form x and y grids and determine grid cell size (requires and checks for regular grid)
        x_vals_unique = np.unique(x_vals)
        dx = x_vals_unique[1] - x_vals_unique[0]
        assert np.allclose(np.diff(x_vals_unique), dx)
        y_vals_unique = np.unique(y_vals)
        dy = y_vals_unique[1] - y_vals_unique[0]
        dxdy = dx * dy
        assert np.allclose(np.diff(y_vals_unique), dy)
        x_grid, y_grid = np.meshgrid(x_vals_unique, y_vals_unique)

        # Grid posterior
        post_grid = scipy.interpolate.griddata((x_vals, y_vals), post, (x_grid, y_grid), fill_value=0)

        # Convert to confidence
        conf_grid = gaussian_cl_likelihood.python.posteriors.posterior_grid_to_confidence_levels(post_grid, dxdy)

        # Meaure area
        cf_areas[:, nbin_idx] = np.count_nonzero((conf_grid[..., np.newaxis] < contour_levels[1:]), axis=(0, 1)) * dxdy

    print('Correlation function: done              ')
    print(cl_areas)
    # Normalise areas and take average across all sigmas
    cl_areas /= np.amin(cl_areas, axis=1)[:, np.newaxis]
    cf_areas /= np.amin(cf_areas, axis=1)[:, np.newaxis]
    cl_areas_avg = np.mean(cl_areas, axis=0)
    cf_areas_avg = np.mean(cf_areas, axis=0)

    # Plot the results
    print('Plotting')
    plt.rcParams.update({'font.size': 13})
    _, ax = plt.subplots(ncols=2, figsize=(12.8, 5))
    plt.subplots_adjust(wspace=0.1, left=.08, right=.99, top=.87)

    leg_label = f'{contour_levels_sig[0]:.0f}\u2013${contour_levels_sig[-1]:.0f} \\sigma$ average'
    ax[0].plot(n_bps, cl_areas_avg, lw=2, label=leg_label)
    ax[1].plot(n_theta_bins, cf_areas_avg, lw=2)

    # Add lines at y=1
    ax[0].axhline(y=1, ls='--', lw=.5, c='k', alpha=.5)
    ax[1].axhline(y=1, ls='--', lw=.5, c='k', alpha=.5)

    ax[1].set_ylim((0.8, 5))

    # Axis labels
    ax[0].set_xlabel('Number of bandpowers')
    ax[1].set_xlabel(r'Number of $\theta$ bins')
    ax[0].set_ylabel('Area inside contour in $w_0$\u2013$w_a$ plane\n(normalised)')

    # Panel labels
    for a, label in zip(ax, ['Power spectrum', 'Correlation function']):
        a.annotate(label, xy=(0.95, 0.95), xycoords='axes fraction', ha='right', va='top', size=14)

    # Legend
    ax[0].legend(loc='lower center', bbox_to_anchor=(1.03, 1.03))

    if save_path is not None:
        plt.savefig(save_path)
        print('Saved ' + save_path)
    else:
        plt.show()


def prepare_width_vs_nbin_grid(cl_like_filemask, cf_like_filemask, contour_levels_sig, n_bps, n_theta_bins, params,
                               lmaxes, theta_min_degs, data_save_path):
    """
    Prepare the width-vs-nbin grid plot, then save the data to file for fast plotting using ``plot_width_vs_nbin_grid``.

    Args:
        cl_like_filemask (str): Path to power spectrum likelihood files as output by
                                ``loop_likelihood_nbin.like_bp_gauss_loop_nbin``, with placeholders for ``{param}``,
                                ``{lmax}`` and ``{n_bp}``.
        cf_like_filemask (str): Path to correlation function likelihood files as output by
                                ``loop_likelihood_nbin.like_cf_gauss_loop_nbin``, with placeholders for ``{param}``,
                                ``{theta_min_deg}`` and ``{n_bin}``.
        contour_levels_sig (list): List or other sequence of sigma confidence levels over which to average the contour
                                   area; can be integers e.g. ``[1, 2, 3]`` or not e.g. ``np.arange(1, 3, 100)``.
        n_bps (list): list or other sequence of numbers of bandpowers over which to iterate.
        n_theta_bins (list): List or other sequence of numbers of theta bins over which to iterate.
        params (list): List of cosmological parameter labels to iterate over as the ``{param}`` argument in
                       ``cl_like_filemask`` and ``cf_like_filemask``.
        lmaxes (list): List of lmax values to iterate over as the ``{lmax}`` argument in ``cl_like_filemask``.
        theta_min_degs (list): List of theta_min values in degrees to iterate over as the ``{theta_min_deg}`` argument
                               in ``{cf_like_filemask}``.
        data_save_path (str): Path to save intermediate data to pass to ``plot_width_vs_nbin_grid``.
    """

    # Calculate sigma levels in probability
    conf_levels = [scipy.special.erf(sig_level / np.sqrt(2)) for sig_level in contour_levels_sig]

    # Calculate other fixed quantities
    n_conf_levels = len(conf_levels)
    n_param = len(params)
    n_n_bp = len(n_bps)
    n_n_tbin = len(n_theta_bins)
    n_lmax = len(lmaxes)

    # Create arrays to store the results: first axis is parameter, second is lmax/tmin, third is number of bins,
    # fourth is contour level
    cl_widths = np.full((n_param, n_lmax, n_n_bp, n_conf_levels), np.nan)
    cf_widths = np.full((n_param, n_lmax, n_n_tbin, n_conf_levels), np.nan)

    # Loop over rows: parameters
    for param_idx, param in enumerate(params):

        # Loop over columns: power spectrum, correlation function
        for cl_or_cf in ['cl', 'cf']:
            assert cl_or_cf in ('cl', 'cf') # sanity check, after which can safely assume that not cl implies cf

            # Array to hold results: first axis is lmax, second is number of bins, third is contour level
            nbins = n_bps if cl_or_cf == 'cl' else n_theta_bins

            # Loop over lmaxes and numbers of bandpowers
            for lmax_idx, (lmax, tmin) in enumerate(zip(lmaxes, theta_min_degs)):
                for nbin_idx, nbin in enumerate(nbins):
                    if cl_or_cf == 'cl':
                        print(f'{param} power spectrum: lmax = {lmax}, n_bp = {nbin}     ', end='\r')
                    else:
                        print(f'{param} correlation function: tmin = {tmin}, nbin = {nbin}     ', end='\r')

                    # Load log-likelihood
                    if cl_or_cf == 'cl':
                        log_like_path = cl_like_filemask.format(param=param, lmax=lmax, n_bp=nbin)
                    else:
                        log_like_path = cf_like_filemask.format(param=param, theta_min_deg=tmin, n_bin=nbin)
                    x_vals, log_like = np.loadtxt(log_like_path, unpack=True)

                    # Convert log-likelihood to unnormalised posterior (flat prior)
                    # while aiming to prevent over/underflows
                    log_like = log_like - np.amax(log_like)
                    post = np.exp(log_like)

                    # Form x grid and determine grid cell size (requires and checks for regular grid)
                    x_grid = np.unique(x_vals)
                    dx = x_grid[1] - x_grid[0]
                    assert np.allclose(np.diff(x_grid), dx)

                    # Grid posterior
                    post_grid = scipy.interpolate.griddata((x_vals), post, (x_grid), fill_value=0)

                    # Interpolate to smooth
                    f = scipy.interpolate.interp1d(x_grid, post_grid, kind='linear')
                    x_grid = np.linspace(x_grid[0], x_grid[-1], int(1e4))
                    post_grid = f(x_grid)
                    dx = x_grid[1] - x_grid[0]

                    # Normalise
                    post_grid /= np.sum(post_grid) * dx
                    assert np.isclose(np.sum(post_grid) * dx, 1)

                    # Convert to confidence
                    conf_grid = gaussian_cl_likelihood.python.posteriors.posterior_grid_to_confidence_levels(post_grid,
                                                                                                             dx)

                    # Measure widths
                    widths = np.count_nonzero((conf_grid[..., np.newaxis] < conf_levels), axis=0) * dx
                    if cl_or_cf == 'cl':
                        cl_widths[param_idx, lmax_idx, nbin_idx, :] = widths
                    else:
                        cf_widths[param_idx, lmax_idx, nbin_idx, :] = widths

            if cl_or_cf == 'cl':
                assert np.all(np.isfinite(cl_widths[param_idx]))
                print(f'{param} power spectrum: done                  ')
            else:
                assert np.all(np.isfinite(cf_widths[param_idx]))
                print(f'{param} correlation function: done                  ')

    # Normalise and average across all sigmas
    assert np.all(np.isfinite(cl_widths))
    assert np.all(np.isfinite(cf_widths))
    cl_widths /= np.amin(cl_widths, axis=2)[:, :, np.newaxis, :]
    cf_widths /= np.amin(cf_widths, axis=2)[:, :, np.newaxis, :]
    cl_widths_avg = np.mean(cl_widths, axis=3)
    cf_widths_avg = np.mean(cf_widths, axis=3)
    assert cl_widths_avg.shape == (n_param, n_lmax, n_n_bp)
    assert cf_widths_avg.shape == (n_param, n_lmax, n_n_tbin)
    assert np.all(np.isfinite(cl_widths))
    assert np.all(np.isfinite(cf_widths))

    # Save to disk
    print('Saving...', end='\r')
    header = (f'Intermediate output from {__file__}.prepare_width_vs_nbin_grid for input '
              f'cl_like_filemask = {cl_like_filemask}, cf_like_filemask = {cf_like_filemask}, '
              f'contour_levels_sig = {contour_levels_sig}, n_bps = {n_bps}, n_theta_bins = {n_theta_bins}, '
              f'params = {params}, lmaxes = {lmaxes}, theta_min_degs = {theta_min_degs}, at {time.strftime("%c")}')
    np.savez_compressed(data_save_path, cl_widths_avg=cl_widths_avg, cf_widths_avg=cf_widths_avg, params=params,
                        lmaxes=lmaxes, tmins=theta_min_degs, n_bps=n_bps, n_tbins=n_theta_bins, header=header)
    print('Saved ' + data_save_path)


def plot_width_vs_nbin_grid(data_path, param_labels, plot_save_path=None):
    """
    Plot grid of single-parameter error against number of angular bins for all parameters and lmax/theta_min values,
    using data prepared with ``prepare_width_vs_nbin_grid``.

    Args:
        data_path (str): Path to data output by ``prepare_width_vs_nbin_grid``.
        param_labels (dict): Dictionary of latex-formatted parameter names corresponding to each parameter ID used in
                             the ``{params}`` argument to ``prepare_width_vs_nbin_grid``, excluding the dollar signs
                             used to indicate maths mode. For example, if omega_m is referred to in likelihood filenames
                             as `omm`, then its entry might be ``{ ... 'omm': 'omega_\\mathrm{m}' ...}``.
        plot_save_path (str, optional): Path to save plot. If not supplied, plot is displayed.
    """

    # Load plot data
    print('Loading')
    with np.load(data_path) as data:
        cl_widths_avg = data['cl_widths_avg']
        cf_widths_avg = data['cf_widths_avg']
        params = data['params']
        lmaxes = data['lmaxes']
        tmins = data['tmins']
        n_bps = data['n_bps']
        n_tbins = data['n_tbins']

    # Derived params
    n_param = len(params)
    n_lmaxes = len(lmaxes)

    # Create grid to plot on
    print('Plotting')
    plt.rcParams.update({'font.size': 13})
    fig, ax = plt.subplots(nrows=n_param, ncols=2, figsize=(12.8, 15), sharex='col')
    plt.subplots_adjust(left=.10, right=.99, bottom=.05, top=.94, wspace=.07, hspace=.1)

    # Loop over columns: power spectrum, correlation function
    cols_widths_avg = [cl_widths_avg, cf_widths_avg]
    cols_labels = ['cl', 'cf']
    cols_nbins = [n_bps, n_tbins]
    for col_idx, (col_widths_avg, col_label, col_nbins), in enumerate(zip(cols_widths_avg, cols_labels, cols_nbins)):

        # Loop over rows: parameters
        for row_idx, panel_widths_avg in enumerate(col_widths_avg):

            # Plot
            for lmax, tmin, lmax_widths in zip(lmaxes, tmins, panel_widths_avg):
                label = (f'$\\ell_\\mathrm{{max}} = {lmax}$' if col_label == 'cl'
                         else f'$\\theta_\\mathrm{{min}} = {tmin}$ deg')
                ax[row_idx, col_idx].plot(col_nbins, lmax_widths, label=label)

            ax[row_idx, col_idx].axhline(y=1, ls='--', c='k', lw=.5, alpha=.5)

    # Shared axis labels
    big_ax = fig.add_subplot(frameon=False)
    big_ax.tick_params(labelcolor='none', bottom=False, left=False)
    big_ax.set_xlabel('Number of angular bins', labelpad=15)
    big_ax.set_ylabel('Single-parameter error bar width (normalised)', labelpad=50)

    # Row labels (parameters)
    for param_idx, param in enumerate(params):
        ax[param_idx, 0].annotate(f'${param_labels[param]}$', xy=(-0.18, 0.5), xycoords='axes fraction', fontsize=14)

    # Column labels
    for top_panel, col_label in zip(ax[0], ['Power spectrum', 'Correlation function']):
        top_panel.annotate(col_label, xy=(0.5, 1.4), xycoords='axes fraction', ha='center', fontsize=14)

    # Legends
    for top_panel in ax[0]:
        top_panel.legend(loc='lower center', bbox_to_anchor=(.5, 1.05), ncol=n_lmaxes, columnspacing=0.6,
                         handlelength=1.5, handletextpad=0.3)

    # Limits
    for cf_panel in ax[:, 1]:
        cf_panel.set_ylim(0.8, 5)

    # Save or show
    if plot_save_path is not None:
        plt.savefig(plot_save_path)
        print('Saved ' + plot_save_path)
    else:
        plt.show()


def area_vs_nbin_fsky_inka(inka_like_filemask, fsky_like_filemask, contour_levels_sig, n_bps, plot_save_path=None):
    """
    Plot w0-wa contour area against number of bandpowers for the cut-sky power spectrum using the improved NKA and fsky
    approximation side-by-side.

    Args:
        inka_like_filemask (str): Path to log-likelihood files generated with the improved NKA method using
                                  ``loop_likelihood_nbin.like_bp_gauss_mix_loop_nbin``, with ``{n_bp}`` placeholder.
        fsky_like_filemask (str): Path to log-likelihood files generated with the fsky approximation using
                                  ``loop_likelihood_nbin.like_bp_gauss_loop_nbin``, with ``{n_bp}`` placeholder.
        contour_levels_sig (list): List or other sequence of sigma confidence levels over which to average the contour
                                   area; can be integers e.g. ``[1, 2, 3]`` or not e.g. ``np.arange(1, 3, 100)``.
        n_bps (list): list or other sequence of numbers of bandpowers over which to iterate.
        plot_save_path (str, optional): Path to save plot. If not supplied, plot will be displayed.
    """

    # Calculate contour levels in probability
    contour_levels = [0.] + [scipy.special.erf(contour_level / np.sqrt(2)) for contour_level in contour_levels_sig]

    # Arrays to hold results: first axis is contour level, second is number of bins
    n_contour_levels = len(contour_levels_sig)
    n_n_bp = len(n_bps)
    inka_areas = np.full((n_contour_levels, n_n_bp), np.nan)
    fsky_areas = np.full((n_contour_levels, n_n_bp), np.nan)

    # Loop over numbers of bandpowers
    for input_path, areas in zip([inka_like_filemask, fsky_like_filemask], [inka_areas, fsky_areas]):
        for n_bp_idx, n_bp in enumerate(n_bps):
            print(f'n_bp = {n_bp}', end='\r')

            # Load log-likelihood
            log_like_path = input_path.format(n_bp=n_bp)
            x_vals, y_vals, log_like = np.loadtxt(log_like_path, unpack=True)

            # Convert log-likelihood to unnormalised posterior (flat prior) while aiming to prevent over/underflows
            log_like = log_like - np.amax(log_like) - 0.5 * np.amin(log_like - np.amax(log_like))
            post = np.exp(log_like)

            # Form x and y grids and determine grid cell size (requires and checks for regular grid)
            x_vals_unique = np.unique(x_vals)
            dx = x_vals_unique[1] - x_vals_unique[0]
            assert np.allclose(np.diff(x_vals_unique), dx)
            y_vals_unique = np.unique(y_vals)
            dy = y_vals_unique[1] - y_vals_unique[0]
            dxdy = dx * dy
            assert np.allclose(np.diff(y_vals_unique), dy)
            x_grid, y_grid = np.meshgrid(x_vals_unique, y_vals_unique)

            # Grid posterior and convert to confidence intervals
            post_grid = scipy.interpolate.griddata((x_vals, y_vals), post, (x_grid, y_grid), fill_value=0)

            # Convert to confidence
            conf_grid = gaussian_cl_likelihood.python.posteriors.posterior_grid_to_confidence_levels(post_grid, dxdy)

            # Measure area
            areas[:, n_bp_idx] = np.count_nonzero((conf_grid[..., np.newaxis] < contour_levels[1:]), axis=(0, 1)) * dxdy

        print()

    assert np.all(np.isfinite(inka_areas))
    assert np.all(np.isfinite(fsky_areas))
    print('Done')

    # Normalise areas and take average across all sigmas
    inka_areas /= np.amin(inka_areas, axis=1)[:, np.newaxis]
    fsky_areas /= np.amin(fsky_areas, axis=1)[:, np.newaxis]
    inka_areas_avg = np.mean(inka_areas, axis=0)
    fsky_areas_avg = np.mean(fsky_areas, axis=0)

    # Plot the results
    print('Plotting')
    plt.rcParams.update({'font.size': 13})
    _, ax = plt.subplots(figsize=(12.8, 4.5))
    plt.subplots_adjust(wspace=0.1, left=.29, right=.71, top=.98, bottom=.12)

    ax.plot(n_bps, inka_areas_avg, lw=2, label='Improved NKA')
    ax.plot(n_bps, fsky_areas_avg, lw=2, label=r'$f_\mathrm{sky}$ approximation')

    # Add line at y=1
    ax.axhline(y=1, ls='--', lw=.5, c='k', alpha=.5)

    # Axis labels
    ax.set_xlabel('Number of bandpowers')
    ax.set_ylabel('Area inside contour in $w_0$\u2013$w_a$ plane\n(normalised)')

    # Legend
    leg_title = f'{contour_levels_sig[0]:.0f}\u2013${contour_levels_sig[-1]:.0f} \\sigma$ average'
    ax.legend(loc='upper right', title=leg_title, frameon=False)

    if plot_save_path is not None:
        plt.savefig(plot_save_path)
        print('Saved ' + plot_save_path)
    else:
        plt.show()


def width_vs_nbin_sqrt_lmax(log_like_filemask, contour_levels_sig, n_bps, params, param_labels, lmaxes,
                            plot_save_path=None):
    """
    Plot single-parameter error against number of bandpowers adjusted as sqrt(lmax) for the full-sky power spectrum.

    Args:
        log_like_filemask (str): Path to log-likelihood files as output by
                                 ``loop_likelihood_nbin.like_bp_gauss_loop_nbin``, with placeholders for ``{param}``,
                                 ``{lmax}`` and ``{n_bp}``.
        contour_levels_sig (list): List or other sequence of sigma confidence levels over which to average the contour
                                   area; can be integers e.g. ``[1, 2, 3]`` or not e.g. ``np.arange(1, 3, 100)``.
        n_bps (list): list or other sequence of numbers of bandpowers over which to iterate.
        params (list): 2D nested list of cosmological parameter IDs, which will be used for the ``{param}`` argument to
                       ``log_like_filemask``. The first dimension represents rows on the grid and the second is columns
                       within each row.
        param_labels (dict): Dictionary of the latex-formatted parameter label (excluding dollar signs denoting maths
                             mode) corresponding to each parameter ID appearing in ``params``.
        lmaxes (list): List or other sequence of lmax values over which to iterate.
        plot_save_path (str, optional): Path to save figure. If not supplied, figure will be displayed.
    """

    # Calculate sigma levels in probability
    conf_levels = [scipy.special.erf(sig_level / np.sqrt(2)) for sig_level in contour_levels_sig]

    # Calculate other fixed quantities
    n_conf_levels = len(conf_levels)
    n_row = len(params)
    n_col = len(params[0])
    assert all(len(row) == n_col for row in params)
    n_n_bp = len(n_bps)
    n_lmax = len(lmaxes)

    # Create grid to plot on
    plt.rcParams.update({'font.size': 13})
    fig, ax = plt.subplots(nrows=n_row, ncols=n_col, figsize=(12.8, 8), sharex='col')
    plt.subplots_adjust(left=.07, right=.99, bottom=.09, top=.92, wspace=.12, hspace=.08)

    # Loop over parameters
    for row_idx, row_params in enumerate(params):
        for col_idx, panel_param in enumerate(row_params):
            print(f'Doing {panel_param}')

            # Array to hold results: first axis is lmax, second is number of bins, third is contour level
            widths = np.full((n_lmax, n_n_bp, n_conf_levels), np.nan)

            # Loop over lmaxes and numbers of bandpowers
            for lmax_idx, lmax in enumerate(lmaxes):
                for n_bp_idx, n_bp in enumerate(n_bps):

                    # Load log-likelihood
                    log_like_path = log_like_filemask.format(param=panel_param, lmax=lmax, n_bp=n_bp)
                    x_vals, log_like = np.loadtxt(log_like_path, unpack=True)

                    # Convert log-likelihood to unnormalised posterior (flat prior)
                    # while aiming to prevent over/underflows
                    log_like = log_like - np.amax(log_like)
                    post = np.exp(log_like)

                    # Form x grid and determine grid cell size (requires and checks for regular grid)
                    x_grid = np.unique(x_vals)
                    dx = x_grid[1] - x_grid[0]
                    assert np.allclose(np.diff(x_grid), dx)

                    # Grid posterior
                    post_grid = scipy.interpolate.griddata((x_vals), post, (x_grid), fill_value=0)

                    # Interpolate to smooth
                    f = scipy.interpolate.interp1d(x_grid, post_grid, kind='linear')
                    x_grid = np.linspace(x_grid[0], x_grid[-1], int(1e4))
                    post_grid = f(x_grid)
                    dx = x_grid[1] - x_grid[0]

                    # Normalise
                    post_grid /= np.sum(post_grid) * dx
                    assert np.isclose(np.sum(post_grid) * dx, 1)

                    # Convert to confidence
                    conf_grid = gaussian_cl_likelihood.python.posteriors.posterior_grid_to_confidence_levels(post_grid,
                                                                                                             dx)

                    # Measure widths
                    widths[lmax_idx, n_bp_idx, :] = np.count_nonzero((conf_grid[..., np.newaxis] < conf_levels),
                                                                     axis=0) * dx

            # Normalise and average across all sigmas
            assert np.all(np.isfinite(widths))
            widths /= np.amin(widths, axis=1)[:, np.newaxis, :]
            widths_avg = np.mean(widths, axis=2)
            assert widths_avg.shape == (n_lmax, n_n_bp)

            # Plot
            for lmax, lmax_widths in zip(lmaxes, widths_avg):
                label = f'$\\ell_\\mathrm{{max}} = {lmax}$'
                plot_x = n_bps * np.sqrt(2000 / lmax)
                ax[row_idx, col_idx].plot(plot_x, lmax_widths, label=label)

            ax[row_idx, col_idx].axhline(y=1, ls='--', c='k', lw=.5, alpha=.5)

    # Shared axis labels
    big_ax = fig.add_subplot(frameon=False)
    big_ax.tick_params(labelcolor='none', bottom=False, left=False)
    big_ax.set_xlabel(r'Number of bandpowers $\times$  $\left[ 2000 ~ / ~ \ell_\mathrm{max} \right]^{1/2}$',
                      labelpad=15)
    big_ax.set_ylabel('Single-parameter error bar width (normalised)', labelpad=20)

    # Panel labels
    for row_idx, row_params in enumerate(params):
        for col_idx, panel_param in enumerate(row_params):
            ax[row_idx, col_idx].annotate(f'${param_labels[panel_param]}$', xy=(.95, .9), xycoords='axes fraction',
                                          ha='right', va='top', fontsize=14)

    # Shared legend
    leg_handles, leg_labels = ax[0, 0].get_legend_handles_labels()
    big_ax.legend(leg_handles, leg_labels, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=n_lmax)

    if plot_save_path is not None:
        plt.savefig(plot_save_path)
        print('Saved ' + plot_save_path)
    else:
        plt.show()


def width_vs_nbin_noise(cl_like_filemask, cf_like_filemask, contour_levels_sig, n_bps, n_theta_bins,
                        plot_save_path=None):
    """
    Plot single-parameter error against number of angular bins for three different noise levels on each panel, with
    three panels: power spectrum with x100/x0.01 noise on top, power spectrum with x2/x0.5 noise on lower left, and
    correlation function with x2/x0.5 noise on lower right.

    Args:
        cl_like_filemask (str): Path to power spectrum log-likelihood files as output by
                                ``loop_likelihood_nbin.like_bp_gauss_loop_nbin``, with placeholders for
                                ``{noise_level}`` and ``{n_bp}``. Values for ``{noise_level}`` are fixed to [0.01, 0.5,
                                1, 2, 100].
        cf_like_filemask (str): Path to correlation function log-likelihood files as output by
                                ``loop_likelihood_nbin.like_cf_gauss_loop_nbin``, with placeholders for
                                ``{noise_level}`` and ``{n_bin}``. Values for ``{noise_level}`` are fixed to [0.5, 1,
                                2].
        contour_levels_sig (list): List or other sequence of sigma confidence levels over which to average the contour
                                   area; can be integers e.g. ``[1, 2, 3]`` or not e.g. ``np.arange(1, 3, 100)``.
        n_bps (list): list or other sequence of numbers of bandpowers over which to iterate.
        n_theta_bins (list): List or other sequence of numbers of theta bins over which to iterate.
        plot_save_path (str, optional): Path to save figure, if supplied. If not supplied, figure will be displayed.
    """

    # Fixed noise levels
    n_noise_levels = 3
    main_noise_levels = [1, 2, 0.5]
    top_noise_levels = [1, 100, 0.01]

    # Calculate sigma levels in probability
    conf_levels = [scipy.special.erf(sig_level / np.sqrt(2)) for sig_level in contour_levels_sig]

    # Array to hold results: first axis is noise level, second is number of bandpowers, third is contour level
    n_n_bp = len(n_bps)
    n_conf_levels = len(conf_levels)
    top_cl_widths = np.full((n_noise_levels, n_n_bp, n_conf_levels), np.nan)
    cl_widths = np.full((n_noise_levels, n_n_bp, n_conf_levels), np.nan)

    # Top: Loop over noise levels and numbers of bandpowers
    for noise_level_idx, noise_level in enumerate(top_noise_levels):
        for n_bp_idx, n_bp in enumerate(n_bps):
            print(f'noise level = {noise_level}, n_bp = {n_bp}     ', end='\r')

            # Load log-likelihood
            log_like_path = cl_like_filemask.format(noise_level=noise_level, n_bp=n_bp)
            x_vals, log_like = np.loadtxt(log_like_path, unpack=True)

            # Convert log-likelihood to unnormalised posterior (flat prior) while aiming to prevent over/underflows
            log_like = log_like - np.amax(log_like)
            post = np.exp(log_like)

            # Form x grid and determine grid cell size (requires and checks for regular grid)
            x_grid = np.unique(x_vals)
            dx = x_grid[1] - x_grid[0]
            assert np.allclose(np.diff(x_grid), dx)

            # Grid posterior
            post_grid = scipy.interpolate.griddata((x_vals), post, (x_grid), fill_value=0)

            # Interpolate to smooth
            interp = scipy.interpolate.interp1d(x_grid, post_grid, kind='linear')
            x_grid = np.linspace(x_grid[0], x_grid[-1], int(1e4))
            post_grid = interp(x_grid)
            dx = x_grid[1] - x_grid[0]

            # Normalise
            post_grid /= np.sum(post_grid) * dx
            assert np.isclose(np.sum(post_grid) * dx, 1)

            # Convert to confidence
            conf_grid = gaussian_cl_likelihood.python.posteriors.posterior_grid_to_confidence_levels(post_grid, dx)

            # Measure widths
            top_cl_widths[noise_level_idx, n_bp_idx, :] = np.count_nonzero((conf_grid[..., np.newaxis] < conf_levels),
                                                                           axis=0) * dx

    assert np.all(np.isfinite(top_cl_widths))
    print()

    # Lower left: Loop over noise levels and numbers of bandpowers
    for noise_level_idx, noise_level in enumerate(main_noise_levels):
        for n_bp_idx, n_bp in enumerate(n_bps):
            print(f'noise level = {noise_level}, n_bp = {n_bp}     ', end='\r')

            # Load log-likelihood
            log_like_path = cl_like_filemask.format(noise_level=noise_level, n_bp=n_bp)
            x_vals, log_like = np.loadtxt(log_like_path, unpack=True)

            # Convert log-likelihood to unnormalised posterior (flat prior) while aiming to prevent over/underflows
            log_like = log_like - np.amax(log_like)
            post = np.exp(log_like)

            # Form x grid and determine grid cell size (requires and checks for regular grid)
            x_grid = np.unique(x_vals)
            dx = x_grid[1] - x_grid[0]
            assert np.allclose(np.diff(x_grid), dx)

            # Grid posterior
            post_grid = scipy.interpolate.griddata((x_vals), post, (x_grid), fill_value=0)

            # Interpolate to smooth
            interp = scipy.interpolate.interp1d(x_grid, post_grid, kind='linear')
            x_grid = np.linspace(x_grid[0], x_grid[-1], int(1e4))
            post_grid = interp(x_grid)
            dx = x_grid[1] - x_grid[0]

            # Normalise
            post_grid /= np.sum(post_grid) * dx
            assert np.isclose(np.sum(post_grid) * dx, 1)

            # Convert to confidence
            conf_grid = gaussian_cl_likelihood.python.posteriors.posterior_grid_to_confidence_levels(post_grid, dx)

            # Measure widths
            cl_widths[noise_level_idx, n_bp_idx, :] = np.count_nonzero((conf_grid[..., np.newaxis] < conf_levels),
                                                                       axis=0) * dx

    assert np.all(np.isfinite(cl_widths))
    print()

    # Array to hold CF results: first axis is noise level, second is number of theta bins, third is contour level
    n_nbin = len(n_theta_bins)
    cf_widths = np.full((n_noise_levels, n_nbin, n_conf_levels), np.nan)

    # Correlation function: Loop over noise levels and numbers of theta bins
    for noise_level_idx, noise_level in enumerate(main_noise_levels):
        for nbin_idx, nbin in enumerate(n_theta_bins):
            print(f'noise_level = {noise_level}, nbin = {nbin}     ', end='\r')

            # Load log-likelihood
            log_like_path = cf_like_filemask.format(noise_level=noise_level, n_bin=nbin)
            x_vals, log_like = np.loadtxt(log_like_path, unpack=True)

            # Convert log-likelihood to unnormalised posterior (flat prior) while aiming to prevent over/underflows
            log_like = log_like - np.amax(log_like)
            post = np.exp(log_like)

            # Form x grid and determine grid cell size (requires and checks for regular grid)
            x_grid = np.unique(x_vals)
            dx = x_grid[1] - x_grid[0]
            assert np.allclose(np.diff(x_grid), dx)

            # Grid posterior
            post_grid = scipy.interpolate.griddata((x_vals), post, (x_grid), fill_value=0)

            # Interpolate to smooth
            interp = scipy.interpolate.interp1d(x_grid, post_grid, kind='linear')
            x_grid = np.linspace(x_grid[0], x_grid[-1], int(1e4))
            post_grid = interp(x_grid)
            dx = x_grid[1] - x_grid[0]

            # Normalise
            post_grid /= np.sum(post_grid) * dx
            assert np.isclose(np.sum(post_grid) * dx, 1)

            # Convert to confidence
            conf_grid = gaussian_cl_likelihood.python.posteriors.posterior_grid_to_confidence_levels(post_grid, dx)

            # Measure widths
            cf_widths[noise_level_idx, nbin_idx, :] = np.count_nonzero((conf_grid[..., np.newaxis] < conf_levels),
                                                                       axis=0) * dx

    assert np.all(np.isfinite(cf_widths))
    print()

    # Normalise and average the results
    top_cl_widths /= np.mean(top_cl_widths[:, -8:, :], axis=1)[:, np.newaxis]
    cl_widths /= np.mean(cl_widths[:, -8:, :], axis=1)[:, np.newaxis]
    cf_widths /= np.amin(cf_widths, axis=1)[:, np.newaxis]
    top_cl_widths_avg = np.mean(top_cl_widths, axis=2)
    cl_widths_avg = np.mean(cl_widths, axis=2)
    cf_widths_avg = np.mean(cf_widths, axis=2)

    # Prepare plot
    plt.rcParams.update({'font.size': 13, 'lines.linewidth': 2})
    fig = plt.figure(figsize=(12.8, 9))
    gs = matplotlib.gridspec.GridSpec(4, 4)
    ax = [[fig.add_subplot(gs[:2, 1:3])], [fig.add_subplot(gs[2:, :2]), fig.add_subplot(gs[2:, 2:])]]
    plt.subplots_adjust(wspace=.25, hspace=.5, left=.06, right=.99, top=.99, bottom=.06)

    # Plot the results
    noise_label = lambda noise_level: 'Baseline' if noise_level == 1 else f'$\\times${noise_level} noise'
    for noise_level, lmax_widths in zip(top_noise_levels, top_cl_widths_avg):
        ax[0][0].plot(n_bps, lmax_widths, label=noise_label(noise_level))
    for noise_level, lmax_widths in zip(main_noise_levels, cl_widths_avg):
        ax[1][0].plot(n_bps, lmax_widths, label=noise_label(noise_level))
    for noise_level, lmax_widths in zip(main_noise_levels, cf_widths_avg):
        ax[1][1].plot(n_theta_bins, lmax_widths, label=noise_label(noise_level))

    # Limits
    ax[0][0].set_ylim(0.99, 1.21)
    ax[1][0].set_ylim(0.99, 1.18)
    ax[1][1].set_ylim(0.8, 5)

    # Axis labels
    ax[0][0].set_xlabel('Number of bandpowers')
    ax[1][0].set_xlabel('Number of bandpowers')
    ax[1][1].set_xlabel(r'Number of $\theta$ bins')
    ax[0][0].set_ylabel('$w_0$ error width (normalised)', labelpad=10)
    ax[1][0].set_ylabel('$w_0$ error width (normalised)', labelpad=10)

    # Legends
    leg_args = {'frameon': False, 'title_fontsize': 14}
    ax[0][0].legend(title='Power spectrum', **leg_args)
    ax[1][0].legend(title='Power spectrum', **leg_args)
    ax[1][1].legend(title='Correlation function', **leg_args)

    if plot_save_path is not None:
        plt.savefig(plot_save_path)
        print('Saved ' + plot_save_path)
    else:
        plt.show()
