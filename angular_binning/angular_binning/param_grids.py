"""
Functions for pre- and post-processing CosmoSIS grids.
"""

import glob
import os.path
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

import angular_binning.like_bp_gauss as like_bp
import angular_binning.like_cf_gauss as like_cf


def reduced_grid_chains(params, upper_diag, lower_diag, n_chains, output_dir=None, plot=False):
    """
    Generate a custom number of chain text files for input to the list sampler in CosmoSIS, covering a two-dimensional
    grid but only including the range within two specified diagonals for efficiency.

    Args:
        params (dict): Dictionary containing one sub-dictionary for each parameter to be varied. Each subdirectory
                       should contain three items: min (minimum value), max (maximum value) and steps (number of steps
                       including endpoints) - see example below.
        upper_diag (list of tuples): The upper limit of the parameter space to include. Upper and lower diagonal limits
                                     are defined by their endpoints, in order of increasing value of the first
                                     parameter (the parameter appearing first in the argument to ``params``). E.g.
                                     ``[(-1.0, -0.5), (1.0, 0.0)]``. It can be useful to use the ``plot`` option to
                                     visualise this.
        lower_diag (list of tuples): The lower limit of the parameter space to include. See ``upper_diag``.
        n_chains (int): Number of chains to output.
        output_dir (str, optional): Path to output directory where the text files will be saved. If not supplied, no
                                    output is produced, which can be useful when wanting only to visualise the grid
                                    using the ``plot`` option.
        plot (bool, optional): Display the grid points to be included.

    Example for params::

        params = {
            'cosmological_parameters--w': {
                'min': -1.5,
                'max': -0.5,
                'steps': 41
            },
            'cosmological_parameters--wa': {
                'min': -1.0,
                'max': 1.0,
                'steps': 51
            }
        }
    """

    # This function only makes sense for 2 dimensions
    assert len(params) == 2

    # Calculate full ranges of each individual parameter
    param_ranges = [np.linspace(param['min'], param['max'], param['steps']) for param in params.values()]

    # Combine into a flat grid of parameter values
    param_grid = np.stack(np.meshgrid(*param_ranges, indexing='ij'), axis=-1)
    flat_shape = (np.product([param['steps'] for param in params.values()]), len(params))
    param_list = np.reshape(param_grid, flat_shape)

    # Calculate the equations of the upper and lower diagonals, in terms of the slope (m) and intercept (c)
    lo_m = (lower_diag[1][1] - lower_diag[0][1]) / (lower_diag[1][0] - lower_diag[0][0])
    lo_c = lower_diag[0][1] - lo_m * lower_diag[0][0]
    up_m = (upper_diag[1][1] - upper_diag[0][1]) / (upper_diag[1][0] - upper_diag[0][0])
    up_c = upper_diag[0][1] - up_m * upper_diag[0][0]

    # Remove points outside the diagonal lines
    above_lower_diag = param_list[:, 1] > (lo_m * param_list[:, 0] + lo_c)
    below_upper_diag = param_list[:, 1] < (up_m * param_list[:, 0] + up_c)
    param_list = param_list[np.logical_and(above_lower_diag, below_upper_diag), :]
    print(f'Final number of grid points: {param_list.shape[0]}')

    if plot:
        plt.scatter(param_list[:, 0], param_list[:, 1], s=1)
        plt.show()

    if output_dir is None:
        return

    # Split into chains of near-equal size
    chains = np.array_split(param_list, n_chains)

    # Save chains to file
    header = ' '.join(params)
    for i, chain in enumerate(chains):
        chain_path = os.path.join(output_dir, f'chain{i}.txt')
        np.savetxt(chain_path, chain, header=header)
        print(f'Saved {chain_path}')


def single_param_chains(fid_params, param_to_vary, half_range, steps, n_chains, output_dir):
    """
    Generate a custom number of chain text files for input to the list sampler in CosmoSIS, varying a single parameter
    with other parameters held fixed at their fiducial values.

    Args:
        fid_params (dict): Dictionary containing fiducial values for each parameter, using parameter names as used in
                           CosmoSIS, excluding the 'cosmological_parameters--' prefix,
                           e.g. ``{'w': -1.0, 'wa': 0.0, ...}``.
        param_to_vary (str): Name of the parameter to vary.
        half_range (float): Parameter will be varied from its fiducial value +/- ``half_range``.
        steps (int): Number of steps with which to vary the parameter, from (fiducial - ``half_range``) to
                     (fiducial + ``half_range``), including the endpoints.
        n_chains (int): Number of chains to output.
        output_dir (str): Path to output directory where the text files will be saved.
    """

    # Calculate ranges of each individual parameter
    param_ranges = []
    for param, fid_val in fid_params.items():
        if param == param_to_vary:
            param_range = np.linspace(fid_val - half_range, fid_val + half_range, steps)
        else:
            param_range = [fid_val]
        param_ranges.append(param_range)

    # Combine into a flat grid of parameter values
    n_params = len(fid_params)
    param_grid = np.stack(np.meshgrid(*param_ranges, indexing='ij'), axis=-1)
    flat_shape = (steps, n_params)
    param_list = np.reshape(param_grid, flat_shape)

    # Split into chains of near-equal size
    chains = np.array_split(param_list, n_chains)

    # Save chains to file
    params_header = ['cosmological_parameters--' + param_name for param_name in fid_params]
    header = ' '.join(params_header)
    for i, chain in enumerate(chains):
        chain_path = os.path.join(output_dir, f'chain{i}.txt')
        np.savetxt(chain_path, chain, header=header)
        print(f'Saved {chain_path}')


def get_diagonal_params(grid_dir, varied_params, save_path):
    """
    From a two-parameter CosmoSIS grid, export a list of parameters from one corner of the grid (min values of the two
    parameters) to the opposite corner (max values of the two parameters).

    Args:
        grid_dir (str): Path to CosmoSIS grid.
        varied_params (list): List of the two CosmoSIS parameter names whose values are varied across the grid.
        save_path (str): Path to save list of parameter values as a text file.
    """

    assert len(varied_params) == 2
    x = []
    y = []

    # Loop over every input directory
    source_dirs = glob.glob(os.path.join(grid_dir, '_[0-9]*/'))
    n_dirs = len(source_dirs)
    if n_dirs == 0:
        warnings.warn(f'No matching directories. Terminating at {time.strftime("%c")}')
        return
    for source_dir in source_dirs:

        # Extract cosmological parameters
        params = [None, None]
        values_path = os.path.join(source_dir, 'cosmological_parameters/values.txt')
        with open(values_path, encoding='UTF-8') as f:
            for line in f:
                for param_idx, param in enumerate(varied_params):
                    param_str = f'{param} = '
                    if param_str in line:
                        params[param_idx] = float(line[len(param_str):])
        err_str = f'Not all parameters in varied_params found in {values_path}'
        assert np.all([param is not None for param in params]), err_str
        x.append(params[0])
        y.append(params[1])

    # Extract the diagonal
    assert len(x) == len(y)
    x = np.unique(x)
    y = np.unique(y)
    to_export = np.array([[x_i, y_i] for x_i, y_i in zip(x, y)])

    # Save to disk
    param_names = ' '.join(varied_params)
    header = (f'Output from {__file__}.get_diagonal_params for grid_dir = {grid_dir}, at {time.strftime("%c")}\n'
              f'{param_names}')
    np.savetxt(save_path, to_export, header=header)
    print('Saved ' + save_path)


def load_diagonal_shear_cl(grid_dir, diag_params_path, save_path):
    """
    Load the first shear power spectrum for each point in parameter space output by ``get_diagonal_params``, and save
    into a single file.

    Args:
        grid_dir (str): Path to CosmoSIS grid.
        diag_params_path (str): Path to text file listing parameters across the diagonal of the grid, as output by
                                ``get_diagonal_params``.
        save_path (str): Path to save output as a numpy .npz file.
    """

    # Load the diagonal parameters and create arrays to hold Cls and whether each parameter has been matched yet
    diag_params = np.loadtxt(diag_params_path)
    n_samp = len(diag_params)
    matched = [False]*n_samp
    cls_ = [None]*n_samp

    # Iterate over directories matching the mask
    source_dirs = glob.glob(os.path.join(grid_dir, '_[0-9]*/'))
    n_dirs = len(source_dirs)
    for i, source_dir in enumerate(source_dirs):
        print(f'{i + 1} / {n_dirs}', end='\r')

        # Extract parameters
        with open(os.path.join(source_dir, 'cosmological_parameters/values.txt'), encoding='ascii') as f:
            w0 = None
            wa = None
            for line in f:
                if 'w =' in line:
                    w0 = float(line[len('w = '):])
                elif 'wa =' in line:
                    wa = float(line[len('wa = '):])
            assert w0 is not None and wa is not None, f'Parameters missing from {source_dir} values file'

        # If they match a point on the diagonal, store the first shear power spectrum
        for diag_param_idx, diag_param in enumerate(diag_params):
            if np.isclose(w0, diag_param[0]) and np.isclose(wa, diag_param[1]):
                print(f'Matched {(w0, wa)} to {diag_param} in {source_dir}')
                matched[diag_param_idx] = True
                cl = np.loadtxt(f'{source_dir}/shear_cl/bin_1_1.txt')
                cls_[diag_param_idx] = cl
                break

    print()
    assert all(matched), 'Not all values were matched to a directory'

    # Save a file with the parameters and the corresponding power spectra
    w0 = diag_params[:, 0]
    wa = diag_params[:, 1]
    shear_cl_bin_1_1 = np.array(cls_)
    assert np.all(np.isfinite(shear_cl_bin_1_1))
    header = (f'Output from {__file__}.load_diagonal_shear_cl for grid_dir = {grid_dir}, '
              f'diag_params_path = {diag_params_path}, at {time.strftime("%c")}')
    np.savez_compressed(save_path, w0=w0, wa=wa, shear_cl_bin_1_1=shear_cl_bin_1_1, header=header)
    print('Saved ' + save_path)


def load_diagonal_3x2pt_cl(grid_dir, diag_params_path, save_path, lmax, lmin_in, n_zbin):
    """
    Load all 3x2pt power spectra for each point in parameter space output by ``get_diagonal_params``, and save
    into a single file.

    Args:
        grid_dir (str): Path to CosmoSIS grid.
        diag_params_path (str): Path to text file listing parameters across the diagonal of the grid, as output by
                                ``get_diagonal_params``.
        save_path (str): Path to save output as a numpy .npz file.
        lmax (int): Maximum to load.
        lmin_in (int): Minimum l in input.
        n_zbin (int): Number of redshift bins, assuming 1 position and 1 shear field per redshift bin.
    """

    # Load the diagonal parameters and create arrays to hold Cls and whether each parameter has been matched yet
    diag_params = np.loadtxt(diag_params_path)
    n_samp = len(diag_params)
    matched = [False]*n_samp
    theory_cls = [None]*n_samp

    # Iterate over directories matching the mask
    source_dirs = glob.glob(os.path.join(grid_dir, '_[0-9]*/'))
    n_dirs = len(source_dirs)
    for i, source_dir in enumerate(source_dirs):
        print(f'{i + 1} / {n_dirs}', end='\r')

        # Extract parameters
        with open(os.path.join(source_dir, 'cosmological_parameters/values.txt'), encoding='ascii') as f:
            w0 = None
            wa = None
            for line in f:
                if 'w =' in line:
                    w0 = float(line[len('w = '):])
                elif 'wa =' in line:
                    wa = float(line[len('wa = '):])
            assert w0 is not None and wa is not None, f'Parameters missing from {source_dir} values file'

        # If they match a point on the diagonal, load and store power spectra
        for diag_param_idx, diag_param in enumerate(diag_params):
            if np.isclose(w0, diag_param[0]) and np.isclose(wa, diag_param[1]):
                print(f'Matched {(w0, wa)} to {diag_param} in {source_dir}')
                matched[diag_param_idx] = True
                pos_pos_dir = os.path.join(source_dir, 'galaxy_cl/')
                she_she_dir = os.path.join(source_dir, 'shear_cl/')
                pos_she_dir = os.path.join(source_dir, 'galaxy_shear_cl/')
                cls_ = like_bp.load_cls(n_zbin, pos_pos_dir, she_she_dir, pos_she_dir, lmax, lmin_in)
                theory_cls[diag_param_idx] = cls_[:, lmin_in:]
                break

    print()
    assert all(matched), 'Not all values were matched to a directory'

    # Save a file with the parameters and the corresponding power spectra
    w0 = diag_params[:, 0]
    wa = diag_params[:, 1]
    theory_cls = np.array(theory_cls)
    assert np.all(np.isfinite(theory_cls))
    header = (f'Output from {__file__}.load_diagonal_3x2pt_cl for grid_dir = {grid_dir}, '
              f'diag_params_path = {diag_params_path}, lmax = {lmax}, lmin_in = {lmin_in}, n_zbin = {n_zbin}, '
              f'at {time.strftime("%c")}')
    np.savez_compressed(save_path, w0=w0, wa=wa, theory_cls=theory_cls, header=header)
    print('Saved ' + save_path)


def load_diagonal_3x2pt_cl_with_b(grid_dir, diag_params_path, save_path, lmax, lmin_in, n_zbin):
    """
    Load all 3x2pt power spectra, inserting zero B-modes, for each point in parameter space output by
    ``get_diagonal_params``, and save into a single file.

    Args:
        grid_dir (str): Path to CosmoSIS grid.
        diag_params_path (str): Path to text file listing parameters across the diagonal of the grid, as output by
                                ``get_diagonal_params``.
        save_path (str): Path to save output as a numpy .npz file.
        lmax (int): Maximum to load.
        lmin_in (int): Minimum l in input.
        n_zbin (int): Number of redshift bins, assuming 1 position and 1 shear field per redshift bin.
    """

    # Load the diagonal parameters and create arrays to hold Cls and whether each parameter has been matched yet
    diag_params = np.loadtxt(diag_params_path)
    n_samp = len(diag_params)
    matched = [False]*n_samp
    theory_cls = [None]*n_samp

    # Iterate over directories matching the mask
    source_dirs = glob.glob(os.path.join(grid_dir, '_[0-9]*/'))
    n_dirs = len(source_dirs)
    for i, source_dir in enumerate(source_dirs):
        print(f'{i + 1} / {n_dirs}', end='\r')

        # Extract parameters
        with open(os.path.join(source_dir, 'cosmological_parameters/values.txt'), encoding='ascii') as f:
            w0 = None
            wa = None
            for line in f:
                if 'w =' in line:
                    w0 = float(line[len('w = '):])
                elif 'wa =' in line:
                    wa = float(line[len('wa = '):])
            assert w0 is not None and wa is not None, f'Parameters missing from {source_dir} values file'

        # If they match a point on the diagonal, load and store power spectra
        for diag_param_idx, diag_param in enumerate(diag_params):
            if np.isclose(w0, diag_param[0]) and np.isclose(wa, diag_param[1]):
                print(f'Matched {(w0, wa)} to {diag_param} in {source_dir}')
                matched[diag_param_idx] = True
                pos_pos_dir = os.path.join(source_dir, 'galaxy_cl/')
                she_she_dir = os.path.join(source_dir, 'shear_cl/')
                pos_she_dir = os.path.join(source_dir, 'galaxy_shear_cl/')
                cls_, _ = like_cf.load_cls_zerob(n_zbin, pos_pos_dir, she_she_dir, pos_she_dir, lmax, lmin_in)
                theory_cls[diag_param_idx] = cls_[:, lmin_in:]
                break

    print()
    assert all(matched), 'Not all values were matched to a directory'

    # Save a file with the parameters and the corresponding power spectra
    w0 = diag_params[:, 0]
    wa = diag_params[:, 1]
    theory_cls = np.array(theory_cls)
    assert np.all(np.isfinite(theory_cls))
    header = (f'Output from {__file__}.load_diagonal_3x2pt_cl_with_b for grid_dir = {grid_dir}, '
              f'diag_params_path = {diag_params_path}, lmax = {lmax}, lmin_in = {lmin_in}, n_zbin = {n_zbin}, '
              f'at {time.strftime("%c")}')
    np.savez_compressed(save_path, w0=w0, wa=wa, theory_cls=theory_cls, header=header)
    print('Saved ' + save_path)
