"""
Functions for plotting posterior distributions.
"""

import gaussian_cl_likelihood.python.posteriors # https://github.com/robinupham/gaussian_cl_likelihood
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.ndimage
import scipy.special
import scipy.stats
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['mathtext.fontset'] = 'cm'

def cl_posts(log_like_filemask, contour_levels_sig, n_bps, colours, linestyles, ellipse_check=False,
             plot_save_path=None):
    """
    Plot w0-wa joint posteriors from the full-sky power spectrum for different numbers of bandpowers.

    Args:
        log_like_filemask (str): Path to log-likelihood files as output by
                                 ``loop_likelihood_nbin.like_bp_gauss_loop_nbin``, with placeholder for ``{n_bp}``.
        contour_levels_sig (list): List or other sequence of integer contour sigma levels to plot.
        n_bps (list): 2D nested list of numbers of bandpowers to plot. The top-level list defines the panels; the inner
                      lists are the different numbers of bandpowers to plot within each panel. For example,
                      ``[[30, 1], [30, 5], [30, 10]]`` will produce three panels showing 1, 5, and 10 bandpowers,
                      with all panels also showing 30 bandpowers. There must be the same number of numbers of bandpowers
                      within each panel.
        colours (list): List of matplotlib colours, corresponding to the different numbers of bandpowers within each
                        panel. All panels will use the same colours.
        linestyles (list): Like ``colours``, but matplotlib linestyles.
        ellipse_check (bool, optional): This function uses ellipse-fitting to overcome sampling noise. If
                                        ``ellipse_check`` is set to ``True``, the raw posterior will be plotted as well
                                        as the fitted ellipse, to check the fit. Default ``False``.
        plot_save_path (str, optional): Path to save the figure, if supplied. If not supplied, figure will be displayed.
    """

    # Calculate contour levels in probability
    contour_levels = [0.] + [scipy.special.erf(contour_level / np.sqrt(2)) for contour_level in contour_levels_sig]

    # Prepare plot
    plt.rcParams.update({'font.size': 13})
    _, ax = plt.subplots(ncols=3, figsize=(12.8, 4), sharey=True)
    plt.subplots_adjust(left=0.07, right=.97, wspace=0, bottom=.13, top=.98)

    # Plot each panel at a time
    for panel_idx, (a, panel_n_bps) in enumerate(zip(ax, n_bps)):
        for n_bp, colour, linestyle in zip(panel_n_bps, colours, linestyles):
            print(f'Panel {panel_idx + 1} / {len(n_bps)}, n_bp = {n_bp}')

            # Load log-likelihood
            log_like_path = log_like_filemask.format(n_bp=n_bp)
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
            # JW CHANGED HERE!
            conf_grid = gaussian_cl_likelihood.python.posteriors.posterior_grid_to_confidence_levels(post_grid, dxdy)
            #conf_grid = gaussian_cl_likelihood.python.posteriors.post_to_conf(post_grid, dxdy)

            # Calculate contours
            cont = a.contour(x_grid, y_grid, conf_grid, levels=contour_levels, colors=colour, linestyles=linestyle,
                             linewidths=2.5)

            # Fit ellipse
            for collection in cont.collections:
                paths = collection.get_paths()
                if not paths:
                    continue

                # Find biggest enclosed contour
                path_lengths = [path.vertices.shape[0] for path in paths]
                main_path = paths[np.argmax(path_lengths)]
                path_x = main_path.vertices[:, 0]
                path_y = main_path.vertices[:, 1]

                # Calculate ellipse centre using midpoint of x and y
                centre = ((np.amax(path_x) + np.amin(path_x)) / 2, (np.amax(path_y) + np.amin(path_y)) / 2)

                # Calculate angle using linear regression
                slope, _, _, _, _ = scipy.stats.linregress(path_y, path_x)
                phi = -np.arctan(slope)

                # Calculate ellipse 'height' (major axis) by finding y range and adjusting for angle
                height = np.ptp(path_y) / np.cos(phi)

                # Calculate ellipse 'width' (minor axis) by rotating everything clockwise by phi,
                # then finding range of new x
                width = np.ptp(np.cos(-phi) * path_x - np.sin(-phi) * path_y)

                # Draw the ellipse and hide the original
                fit_ellipse = matplotlib.patches.Ellipse(xy=centre, width=width, height=height, angle=np.rad2deg(phi),
                                                         ec=collection.get_ec()[0], fc='None',
                                                         lw=collection.get_lw()[0], ls=collection.get_ls()[0])
                a.add_patch(fit_ellipse)
                if not ellipse_check:
                    collection.set_visible(False)

    # Limits
    for a in ax:
        #a.set_xlim(-1.01, -0.99)
        a.set_xlim(-1.25, -0.75)
        #a.set_ylim(-0.03, 0.035)
        a.set_ylim(-0.25, 0.25)

    # Axis labels
    for a in ax:
        a.set_xlabel(r'$w_0$')
    ax[0].set_ylabel(r'$w_a$')

    # Legends
    for a, panel_n_bps in zip(ax, n_bps):
        handles = [matplotlib.lines.Line2D([0], [0], lw=2.5, c=c, ls=ls[0]) for c, ls in zip(colours, linestyles)]
        labels = [f'{n_bp} bandpower{"s" if n_bp > 1 else ""}' for n_bp in panel_n_bps]
        a.legend(handles, labels, frameon=False, loc='upper right')

    # Remove overlapping tick labels
    for a in ax[1:]:
        a.set_xticks(a.get_xticks()[1:])

    if plot_save_path is not None:
        plt.savefig(plot_save_path)
        print('Saved ' + plot_save_path)
    else:
        plt.show()


def cl_post(log_like_filemask, contour_levels_sig, bp, colour, linestyle, zrange, lrange, ngals, nside,
            n_bandpowers, obs_type, ellipse_check=False, plot_save_path=None):
    """
    Plot w0-wa joint posteriors from the full-sky power spectrum for a single bandpower.

    Args:
        log_like_filemask (str): Path to log-likelihood files as output by
                                 ``loop_likelihood_nbin.like_bp_gauss_loop_nbin``, with placeholder for ``{n_bp}``.
        contour_levels_sig (list): List or other sequence of integer contour sigma levels to plot.
        bp (float, int): Single bandpower number.
        colour (string): Colour.
        linestyles (list): matplotlib linestyle.
        ellipse_check (bool, optional): This function uses ellipse-fitting to overcome sampling noise. If
                                        ``ellipse_check`` is set to ``True``, the raw posterior will be plotted as well
                                        as the fitted ellipse, to check the fit. Default ``False``.
        plot_save_path (str, optional): Path to save the figure, if supplied. If not supplied, figure will be displayed.
    """

    # Calculate contour levels in probability
    contour_levels = [0.] + [scipy.special.erf(contour_level / np.sqrt(2)) for contour_level in contour_levels_sig]

    # Prepare plot
    plt.rcParams.update({'font.size': 13})
    fig, ax = plt.subplots(figsize=(4.25, 4),)
    plt.subplots_adjust(left=0.175, right=.925, wspace=0, bottom=.15, top=.875)
    plt.title(
        r'${}\leq\ell\leq{}$'.format(lrange[0], lrange[1]) +
        ', NSIDE={}, {} Bandpowers,\nNGal = {}, {}'.format(nside, n_bandpowers, ngals, zrange[0]) +
        r'$\leq z\leq$' + '{}, {}'.format(zrange[1], obs_type), fontsize=12)

    # Plot each panel at a time
    #for panel_idx, (a, panel_n_bps) in enumerate(zip(ax, n_bps)):
    #    for n_bp, colour, linestyle in zip(panel_n_bps, colours, linestyles):
    #        print(f'Panel {panel_idx + 1} / {len(n_bps)}, n_bp = {n_bp}')

    # Load log-likelihood
    log_like_path = log_like_filemask.format(n_bp=bp)
    x_vals, y_vals, log_like = np.loadtxt(log_like_path, unpack=True)

    # Convert log-likelihood to unnormalised posterior (flat prior) while aiming to prevent over/underflows
    #log_like[log_like>0] = -1e8
    log_like = log_like - max(log_like)
    post = np.exp(log_like)

    # Form x and y grids and determine grid cell size (requires and checks for regular grid)
    x_vals_unique = np.unique(x_vals)
    dx = x_vals_unique[1] - x_vals_unique[0]
    #assert np.allclose(np.diff(x_vals_unique), dx)
    y_vals_unique = np.unique(y_vals)
    dy = y_vals_unique[1] - y_vals_unique[0]
    dxdy = dx * dy
    #assert np.allclose(np.diff(y_vals_unique), dy)
    x_grid, y_grid = np.meshgrid(x_vals_unique, y_vals_unique)

    # Grid posterior and convert to confidence intervals
    post_grid = scipy.interpolate.griddata((x_vals, y_vals), post, (x_grid, y_grid), fill_value=0)

    # Convert to confidence
    conf_grid = gaussian_cl_likelihood.python.posteriors.posterior_grid_to_confidence_levels(post_grid, dxdy)

    # Calculate contours![](../../../FINAL_DATA/NONOISE/EQUIPOP/3x2pt/inference_analysis/contours_l100-1500_3X2PT.png)

    cont = ax.contour(x_grid, y_grid, conf_grid, levels=contour_levels, colors=colour, linestyles=linestyle,
                     linewidths=2.5)
    ax.plot(-1,0,marker='x',color='0') #for 2000

    # Limits
    #for a in ax:
    #ax.set_xlim(-1.08, -0.92)
    #ax.set_ylim(-0.175, 0.175)

    #ax.set_xlim(-1.05, -0.95)
    #ax.set_ylim(-0.1, 0.1)
    ax.set_xlim(-1.25, -0.75)
    ax.set_ylim(-0.5, 0.5)
    #ax.set_xlim(-1.25, -0.75)
    #ax.set_ylim(-0.5, 0.5)

    ax.set_xlabel(r'$w_0$')
    ax.set_ylabel(r'$w_a$')
    ax.yaxis.labelpad=-8

    #circle = plt.Circle((-1,0), 0.03, ls=':',color='red')
    #ax.add_patch(circle)
    '''
    ellipse = Ellipse(
        xy=(-1, 0),
        width=0.01,
        height=0.08,
        angle=15,
        facecolor="red",
        edgecolor="red"
    )
    ax.add_patch(ellipse)
    '''
    #plt.show()
    '''
    #print(cont)
    # Fit ellipse
    for collection in cont.collections:
        paths = collection.get_paths()
        if not paths:
            continue

        # Find biggest enclosed contour
        path_lengths = [path.vertices.shape[0] for path in paths]
        main_path = paths[np.argmax(path_lengths)]
        path_x = main_path.vertices[:, 0]
        path_y = main_path.vertices[:, 1]

        # Calculate ellipse centre using midpoint of x and y
        centre = ((np.amax(path_x) + np.amin(path_x)) / 2, (np.amax(path_y) + np.amin(path_y)) / 2)
        #print(centre)
        # Calculate angle using linear regression
        slope, _, _, _, _ = scipy.stats.linregress(path_y, path_x)
        phi = -np.arctan(slope)

        # Calculate ellipse 'height' (major axis) by finding y range and adjusting for angle
        height = np.ptp(path_y) / np.cos(phi)

        # Calculate ellipse 'width' (minor axis) by rotating everything clockwise by phi,
        # then finding range of new x
        width = np.ptp(np.cos(-phi) * path_x - np.sin(-phi) * path_y)

        # Draw the ellipse and hide the original
        fit_ellipse = matplotlib.patches.Ellipse(xy=centre, width=width, height=height, angle=np.rad2deg(phi),
                                                 ec=collection.get_ec()[0], fc='None',
                                                 lw=collection.get_lw()[0], ls=collection.get_ls()[0])
        ax.add_patch(fit_ellipse)
        if not ellipse_check:
            collection.set_visible(False)

    '''
    # Axis labels
    #for a in ax:


    # Legends
    #for a, panel_n_bps in zip(ax, n_bps):
    #    handles = [matplotlib.lines.Line2D([0], [0], lw=2.5, c=c, ls=ls[0]) for c, ls in zip(colours, linestyles)]
    #    labels = [f'{n_bp} bandpower{"s" if n_bp > 1 else ""}' for n_bp in panel_n_bps]
    #ax.legend(frameon=False, loc='upper right')

    # Remove overlapping tick labels
    #for a in ax[1:]:
    #    a.set_xticks(a.get_xticks()[1:])

    if plot_save_path is not None:
        plt.savefig(plot_save_path)
        print('Saved ' + plot_save_path)
    else:
        plt.show()



def get_contours(log_like_filemask, contour_levels_sig, bp, colour, linestyle):
    """
    Plot w0-wa joint posteriors from the full-sky power spectrum for a single bandpower.

    Args:
        log_like_filemask (str): Path to log-likelihood files as output by
                                 ``loop_likelihood_nbin.like_bp_gauss_loop_nbin``, with placeholder for ``{n_bp}``.
        contour_levels_sig (list): List or other sequence of integer contour sigma levels to plot.
        bp (float, int): Single bandpower number.
        colour (string): Colour.
        linestyles (list): matplotlib linestyle.
        ellipse_check (bool, optional): This function uses ellipse-fitting to overcome sampling noise. If
                                        ``ellipse_check`` is set to ``True``, the raw posterior will be plotted as well
                                        as the fitted ellipse, to check the fit. Default ``False``.
        plot_save_path (str, optional): Path to save the figure, if supplied. If not supplied, figure will be displayed.
    """

    # Calculate contour levels in probability
    contour_levels = [0.] + [scipy.special.erf(contour_level / np.sqrt(2)) for contour_level in contour_levels_sig]

    # Load log-likelihood
    log_like_path = log_like_filemask.format(n_bp=bp)
    x_vals, y_vals, log_like = np.loadtxt(log_like_path, unpack=True)

    # Convert log-likelihood to unnormalised posterior (flat prior) while aiming to prevent over/underflows

    log_like = log_like - max(log_like)
    post = np.exp(log_like)

    # Form x and y grids and determine grid cell size (requires and checks for regular grid)
    x_vals_unique = np.unique(x_vals)
    dx = x_vals_unique[1] - x_vals_unique[0]
    #assert np.allclose(np.diff(x_vals_unique), dx)
    y_vals_unique = np.unique(y_vals)
    dy = y_vals_unique[1] - y_vals_unique[0]
    dxdy = dx * dy
    #assert np.allclose(np.diff(y_vals_unique), dy)
    x_grid, y_grid = np.meshgrid(x_vals_unique, y_vals_unique)

    # Grid posterior and convert to confidence intervals
    post_grid = scipy.interpolate.griddata((x_vals, y_vals), post, (x_grid, y_grid), fill_value=0)

    # Convert to confidence
    conf_grid = gaussian_cl_likelihood.python.posteriors.posterior_grid_to_confidence_levels(post_grid, dxdy)

    # Calculate contours

    return x_grid, y_grid, conf_grid, contour_levels, colour, linestyle

def plot_multiple_contours(contour_params, title, labels, plot_save_path, ellipse_check=False):

    """

    Parameters
    ----------
    contour_params (arr):   Array of N contour parameters generated by get_contours(), i.e. [get_contours(N1),
                            get_countours(N2), ..., ...]

    Returns
    -------
    Plot of multiple contours
    """

    no_contours = len(contour_params)

    # Prepare plot
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(figsize=(5, 5),)

    ellipse = Ellipse(
        xy=(-1, 0),
        width=0.0125,
        height=0.2,
        angle=15,
        facecolor="None",
        edgecolor="0",
        linestyle='--',
        linewidth=1.5,
        zorder=100
    )
    #ax.add_patch(ellipse)

    plt.subplots_adjust(left=0.175, right=.925, wspace=0, bottom=.15, top=.875)
    #plt.title(title, fontsize=12)
    colours=[]
    #for contour in range(no_contours):
    for contour in range(no_contours):
        cont = ax.contour(contour_params[contour][0], contour_params[contour][1], contour_params[contour][2],
                          levels=contour_params[contour][3], colors=contour_params[contour][4],
                          linestyles=contour_params[contour][5], linewidths=2.5)
        '''
        # Fit ellipse
        for collection in cont.collections:
            paths = collection.get_paths()
            if not paths:
                continue

            # Find biggest enclosed contour
            path_lengths = [path.vertices.shape[0] for path in paths]
            main_path = paths[np.argmax(path_lengths)]
            path_x = main_path.vertices[:, 0]
            path_y = main_path.vertices[:, 1]

            # Calculate ellipse centre using midpoint of x and y
            centre = ((np.amax(path_x) + np.amin(path_x)) / 2, (np.amax(path_y) + np.amin(path_y)) / 2)
            # print(centre)
            # Calculate angle using linear regression
            slope, _, _, _, _ = scipy.stats.linregress(path_y, path_x)
            phi = -np.arctan(slope)

            # Calculate ellipse 'height' (major axis) by finding y range and adjusting for angle
            height = np.ptp(path_y) / np.cos(phi)

            # Calculate ellipse 'width' (minor axis) by rotating everything clockwise by phi,
            # then finding range of new x
            width = np.ptp(np.cos(-phi) * path_x - np.sin(-phi) * path_y)

            # Draw the ellipse and hide the original
            corr = 1
            corr2 = 1
            if contour == 1:
                #corr = np.sqrt(1.18)
                corr = 1
                corr2 = 1
                print(centre, width, height, np.rad2deg(phi))
            fit_ellipse = matplotlib.patches.Ellipse(xy=centre, width=width*corr*(1/corr2), height=height*corr*corr2, angle=np.rad2deg(phi),
                                                     ec=collection.get_ec()[0], fc='None',
                                                     lw=collection.get_lw()[0], ls=collection.get_ls()[0])
            ax.add_patch(fit_ellipse)
            if not ellipse_check:
                collection.set_visible(False)
        '''
        colours.append(contour_params[contour][4])
        #print(labels[contour])
        #patch = mpatches.Patch(color=contour_params[contour][4], label=labels[contour])
        #label_patches.append(patch)

    #colors = ['black', 'red', 'green']
    lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in colours]
    lines.append(Line2D([0], [0], color='0', marker='x', linewidth=3, linestyle='None'))
    #lines.append(Line2D([0], [0], color='0', linewidth=1.5, linestyle='--'))
    labels.append('Fiducial\nCosmology')
    #labels.append('Euclid-Like FOM')
    ax.plot(-1, 0, marker='x', color='0')
    #ax.plot(-1.0416666666666667, 0.10833333333333334, marker='^', color='0')
    ax.legend(lines, labels, fontsize=12.5, loc='upper right')
    # Limits
    # for a in ax:
    ax.set_xlim(-1.06, -0.94)
    ax.set_ylim(-0.175, 0.174)
    #ax.set_title('10 Equipopulated Bins',fontsize=15)
    #ax.set_xlim(-1.15, -0.85)
    #ax.set_ylim(-0.4, 0.4)
    #ax.set_xlim(-1.2, -0.8)
    #ax.set_ylim(-0.25, 0.25)
    #ax.set_xlim(-1.1, -0.9)
    #ax.set_ylim(-0.3, 0.3)
    ax.set_xlabel(r'$w_0$', fontsize=18)
    ax.set_ylabel(r'$w_a$', fontsize=18)
    ax.yaxis.labelpad = -5
    #plt.legend()
    plt.savefig(plot_save_path)
    print('Saved ' + plot_save_path)



def cf_posts(log_like_filemask, contour_levels_sig, n_bins, colours, linestyles, ellipse_check=False,
             plot_save_path=None):
    """
    Plot w0-wa joint posteriors from the full-sky correlation function for different numbers of theta bins.

    Args:
        log_like_filemask (str): Path to log-likelihood files as output by
                                 ``loop_likelihood_nbin.like_cf_gauss_loop_nbin``, with placeholder for ``{n_bin}``.
        contour_levels_sig (list): List or other sequence of integer contour sigma levels to plot.
        n_bins (list): 2D nested list of numbers of theta bins to plot. The top-level list defines the panels; the
                       inner lists are the different numbers of theta bins to plot within each panel. For example,
                       ``[[30, 5], [30, 10], [30, 20]]`` will produce three panels showing 5, 10, and 20 theta bins,
                       with all panels also showing 30 bins. There must be the same number of numbers of theta bins
                       within each panel.
        colours (list): List of matplotlib colours, corresponding to the different numbers of theta bins within each
                        panel. All panels will use the same colours.
        linestyles (list): Like ``colours``, but matplotlib linestyles.
        ellipse_check (bool, optional): This function uses ellipse-fitting to overcome sampling noise. If
                                        ``ellipse_check`` is set to ``True``, the raw posterior will be plotted as well
                                        as the fitted ellipse, to check the fit. Default ``False``.
        plot_save_path (str, optional): Path to save the figure, if supplied. If not supplied, figure will be displayed.
    """

    # Calculate contour levels in probability
    contour_levels = [0.] + [scipy.special.erf(contour_level / np.sqrt(2)) for contour_level in contour_levels_sig]

    # Prepare plot
    plt.rcParams.update({'font.size': 13})
    _, ax = plt.subplots(ncols=3, figsize=(12.8, 4), sharey=True)
    plt.subplots_adjust(left=0.07, right=.97, wspace=0, bottom=.13, top=.98)

    # Plot each panel at a time
    for panel_idx, (a, panel_n_bins) in enumerate(zip(ax, n_bins)):
        for nbin, colour, linestyle in zip(panel_n_bins, colours, linestyles):
            print(f'Panel {panel_idx + 1} / {len(n_bins)}, nbin = {nbin}')

            # Load log-likelihood
            log_like_path = log_like_filemask.format(n_bin=nbin)
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

            # Calculate contours
            cont = a.contour(x_grid, y_grid, conf_grid, levels=contour_levels, colors=colour, linestyles=linestyle,
                             linewidths=2.5)

            # Fit ellipse
            for collection in cont.collections:
                paths = collection.get_paths()
                if not paths:
                    continue

                # Find biggest enclosed contour
                path_lengths = [path.vertices.shape[0] for path in paths]
                main_path = paths[np.argmax(path_lengths)]
                path_x = main_path.vertices[:, 0]
                path_y = main_path.vertices[:, 1]

                # Calculate ellipse centre using midpoint of x and y
                centre = ((np.amax(path_x) + np.amin(path_x)) / 2, (np.amax(path_y) + np.amin(path_y)) / 2)

                # Calculate angle using linear regression
                slope, _, _, _, _ = scipy.stats.linregress(path_y, path_x)
                phi = -np.arctan(slope)

                # Calculate ellipse 'height' (major axis) by finding y range and adjusting for angle
                height = np.ptp(path_y) / np.cos(phi)

                # Calculate ellipse 'width' (minor axis) by rotating everything clockwise by phi,
                # then finding range of new x
                width = np.ptp(np.cos(-phi) * path_x - np.sin(-phi) * path_y)

                # Draw the ellipse and hide the original
                fit_ellipse = matplotlib.patches.Ellipse(xy=centre, width=width, height=height, angle=np.rad2deg(phi),
                                                         ec=collection.get_color()[0], fc='None',
                                                         lw=collection.get_lw()[0], ls=collection.get_ls()[0])
                a.add_patch(fit_ellipse)
                if not ellipse_check:
                    collection.set_visible(False)

    # Limits
    for a in ax:
        a.set_xlim(-1.01, -0.99)
        a.set_ylim(-0.033, 0.035)

    # Axis labels
    for a in ax:
        a.set_xlabel(r'$w_0$')
    ax[0].set_ylabel(r'$w_a$')

    # Legends
    for a, panel_n_bins in zip(ax, n_bins):
        handles = [matplotlib.lines.Line2D([0], [0], lw=2.5, c=c, ls=ls[0]) for c, ls in zip(colours, linestyles)]
        labels = [f'{nbin} $\\theta$ bins' for nbin in panel_n_bins]
        a.legend(handles, labels, frameon=False, loc='upper right')

    # Remove overlapping tick labels
    for a in ax[1:]:
        a.set_xticks(a.get_xticks()[1:])

    if plot_save_path is not None:
        plt.savefig(plot_save_path)
        print('Saved ' + plot_save_path)
    else:
        plt.show()
