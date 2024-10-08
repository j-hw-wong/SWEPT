"""
Likelihood module to evaluate the joint likelihood of a set of tomographic 3x2pt power spectra, binned into bandpowers,
on the cut sky using a multivariate Gaussian likelihood.

The main functions are setup, which should be called once per analysis, and execute, which is called for every new
point in parameter space.
"""

import os.path
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, '/raid/scratch/wongj/mywork/3x2pt/angular_binning')
sys.path.insert(1, '/raid/scratch/wongj/mywork/3x2pt/gaussian_cl_likelihood')
sys.path.insert(1, '/raid/scratch/wongj/mywork/3x2pt')
import gaussian_cl_likelihood

def mvg_logpdf_fixedcov(x, mean, inv_cov):
    """
    Log-pdf of the multivariate Gaussian distribution where the determinant and inverse of the covariance matrix are
    precomputed and fixed.
    Note that this neglects the additive constant: -0.5 * (len(x) * log(2 * pi) + log_det_cov), because it is
    irrelevant when comparing pdf values with a fixed covariance, but it means that this is not the normalised pdf.

    Args:
        x (1D numpy array): Vector value at which to evaluate the pdf.
        mean (1D numpy array): Mean vector of the multivariate Gaussian distribution.
        inv_cov (2D numpy array): Inverted covariance matrix.

    Returns:
        float: Log-pdf value.
    """
    #print(x)
    #print(mean)
    #print(inv_cov)
    dev = x - mean
    #print(dev)
    return -0.5 * (dev @ inv_cov @ dev)


def is_even(x):
    """
    True if x is even, false otherwise.

    Args:
        x (float): Number to test.

    Returns:
        bool: True if even.
    """
    return x % 2 == 0


def is_odd(x):
    """
    True if x is odd, false otherwise.

    Args:
        x (float): Number to test.

    Returns:
        bool: True if odd.
    """
    return x % 2 == 1


def load_cls(n_zbin, pos_pos_dir, she_she_dir, pos_she_dir, lmax=None, lmin=0):
    """
    Given the number of redshift bins and relevant directories, load power spectra (position, shear, cross) in the
    correct order (diagonal / healpy new=True ordering).
    If lmin is supplied, the output will be padded to begin at l=0.

    Args:
        n_zbin (int): Number of redshift bins.
        pos_pos_dir (str): Path to directory containing position-position power spectra.
        she_she_dir (str): Path to directory containing shear-shear power spectra.
        pos_she_dir (str): Path to directory containing position-shear power spectra.
        lmax (int, optional): Maximum l to load - if not supplied, will load all lines, which requires the individual
                              lmax of each file to be consistent.
        lmin (int, optional): Minimum l supplied. Output will be padded with zeros below this point.

    Returns:
        2D numpy array: All Cls, with different spectra along the first axis and increasing l along the second.
    """

    # Calculate number of fields assuming 1 position field and 1 shear field per redshift bin
    n_field = 2 * n_zbin

    # Load power spectra in diagonal-major order
    spectra = []
    for diag in range(n_field):
        for row in range(n_field - diag):
            col = row + diag

            # Determine whether position-position, shear-shear or position-shear by whether the row and column are even,
            # odd or mixed
            if is_even(row) and is_even(col):
                cl_dir = pos_pos_dir
            elif is_odd(row) and is_odd(col):
                cl_dir = she_she_dir
            else:
                cl_dir = pos_she_dir

            # Extract the bins: for pos-pos and she-she the higher bin index goes first, for pos-she pos goes first
            bins = (row // 2 + 1, col // 2 + 1)
            if cl_dir in (pos_pos_dir, she_she_dir):
                bin1 = max(bins)
                bin2 = min(bins)
            else:
                if is_even(row): # even means pos
                    bin1, bin2 = bins
                else:
                    bin2, bin1 = bins

            cl_path = os.path.join(cl_dir, f'bin_{bin1}_{bin2}.txt')

            # Load with appropriate ell range
            max_rows = None if lmax is None else (lmax - lmin + 1)
            spec = np.concatenate((np.zeros(lmin), np.loadtxt(cl_path, max_rows=max_rows)))
            spectra.append(spec)

    return np.asarray(spectra)


def apply_pixwin_theory_cls(n_zbin, pos_pos_dir, she_she_dir, pos_she_dir, nside, lmin, lmax, pos_pos_save_dir,
                            she_she_save_dir, pos_she_save_dir):
    '''
    In case maps from which Cls are measured are convolved with a Healpix window function (or could be generalised to
    any window/beam function) then we need to modify our theory Cls before calculating a covariance associated with the
    signal
    Args:
        n_zbin:
        pos_pos_dir:
        she_she_dir:
        pos_she_dir:
        nside:
        lmin:
        lmax:
        pos_pos_save_dir:
        she_she_save_dir:
        pos_she_save_dir:

    Returns:

    '''

    for save_dir in [pos_pos_save_dir, she_she_save_dir, pos_she_save_dir]:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    hpx_win = hp.pixwin(nside=nside, pol=True, lmax=lmax)

    for i in range(n_zbin):
        for j in range(n_zbin):

            pos_she_cls = np.loadtxt(pos_she_dir.format(pos_zbin=i + 1, she_zbin=j + 1))
            np.savetxt(pos_she_save_dir.format(pos_zbin=i + 1, she_zbin=j + 1),
                       np.transpose(pos_she_cls * (hpx_win[0][lmin:]) * (hpx_win[1][lmin:])))

            if i >= j:
                pos_pos_cls = np.loadtxt(pos_pos_dir.format(hi_zbin=i + 1,lo_zbin=j + 1))
                np.savetxt(pos_pos_save_dir.format(hi_zbin=i + 1, lo_zbin=j + 1),
                           np.transpose(pos_pos_cls * (hpx_win[0][lmin:]) * (hpx_win[0][lmin:])))

                she_she_cls = np.loadtxt(she_she_dir.format(hi_zbin=i + 1,lo_zbin=j + 1))
                np.savetxt(she_she_save_dir.format(hi_zbin=i + 1,lo_zbin=j + 1),
                           np.transpose(she_she_cls * (hpx_win[1][lmin:]) * (hpx_win[1][lmin:])))


def setup(obs_bp_path, binmixmat_path, mixmats, mix_lmin, cov_path, pos_nl_path, she_nl_path, noise_lmin,
          input_lmax, n_zbin):
    """
    Load and precompute everything that is fixed throughout parameter space. This should be called once per analysis,
    prior to any calls to execute.

    Args:
        obs_bp_path (str): Path to the observed bandpowers, in a numpy .npz file with array name obs_bp and shape
                           (n_spectra, n_bandpowers), with spectra in diagonal-major order.
        binmixmat_path (str): Path to combined mixing and binning matrices, in numpy .npz file with array names
                              (binmix_tt_to_tt, binmix_te_to_te, binmix_ee_to_ee, binmix_bb_to_ee), each with shape
                              (n_bandpower, input_lmax - mix_lmin + 1).
        mixmats (list): List containing the mixing matrices in order [nn-nn, ne-ne, ee-ee, bb-ee] for use in coupling
                        theory spectra
        mix_lmin (int): Minimum l for the theory power in the mixing matrices.
        cov_path (str): Path to precomputed covariance, in numpy .npz file with array name cov, with shape
                        (n_data, n_data) where n_data = n_spectra * n_bandpowers.
        pos_nl_path (str): Path to the unbinned position noise power spectrum, in text file.
        she_nl_path (str): Path to the unbinned shear noise power spectrum, in text file.
        noise_lmin (int): Minimum l in noise power spectra.
        input_lmax (int): Maximum l to include in mixing. Theory and noise power will be truncated above this.
        n_zbin (int): Number of redshift bins. It will be assumed that there is one position field and one shear field
                      per redshift bin.

    Returns:
        dict: Config dictionary to pass to execute.
    """

    # Load observed bandpowers and prepare into a vector
    with np.load(obs_bp_path) as data:
        obs_bp = data['obs_bp']
    n_spec, n_bandpower = obs_bp.shape
    assert n_spec == (2 * n_zbin) * (2 * n_zbin + 1) // 2
    n_data = n_spec * n_bandpower
    obs_bp = np.reshape(obs_bp, n_data)

    # Load covariance
    with np.load(cov_path) as data:
        cov = data['cov']
    assert cov.shape == (n_data, n_data)

    # To look at the diagonal or absolute covariance, useful for debugging.
    # cov = np.diag(np.diag(cov))
    # cov = np.abs(cov)

    # Invert covariance
    inv_cov = np.linalg.inv(cov)

    #Specify mixing matrices
    mixmat_nn_to_nn = mixmats[0]
    mixmat_ne_to_ne = mixmats[1]
    mixmat_ee_to_ee = mixmats[2]
    mixmat_bb_to_ee = mixmats[3]
    #Could e.g assert mixmat shape == binmix shape

    # Load combined mixing and binning matrices
    with np.load(binmixmat_path) as data:
        binmix_nn_to_nn = data['binmix_tt_to_tt']
        binmix_ne_to_ne = data['binmix_te_to_te']
        binmix_ee_to_ee = data['binmix_ee_to_ee']
        binmix_bb_to_ee = data['binmix_bb_to_ee']
    n_cl = binmix_nn_to_nn.shape[1]
    assert binmix_nn_to_nn.shape == (n_bandpower, n_cl)
    assert binmix_ne_to_ne.shape == (n_bandpower, n_cl)
    assert binmix_ee_to_ee.shape == (n_bandpower, n_cl)
    assert binmix_bb_to_ee.shape == (n_bandpower, n_cl)
    mix_lmax = mix_lmin + n_cl - 1

    # Load noise and trim/pad to correct length for input to mixing matrices,
    # truncating power above input_lmax

    zeros_lowl = np.zeros(noise_lmin)
    zeros_hil = np.zeros(max(mix_lmax - input_lmax, 0))

    if pos_nl_path is not None:
        assert she_nl_path is not None
        pos_nl = np.loadtxt(pos_nl_path, max_rows=(input_lmax - noise_lmin + 1))
        she_nl = np.loadtxt(she_nl_path, max_rows=(input_lmax - noise_lmin + 1))

        pos_nl = np.concatenate((zeros_lowl, pos_nl, zeros_hil))[mix_lmin:(mix_lmax + 1)]
        she_nl = np.concatenate((zeros_lowl, she_nl, zeros_hil))[mix_lmin:(mix_lmax + 1)]
        assert pos_nl.shape == (n_cl, )
        assert she_nl.shape == (n_cl, )
    else:
        assert pos_nl_path is None
        assert she_nl_path is None
        pos_nl = None
        she_nl = None
    # Generate a list of spectrum types (NN, EE or NE) in the correct (diagonal) order, so that we know which mixing
    # matrix/matrices to apply
    fields = [field for _ in range(n_zbin) for field in ('N', 'E')]
    n_field = len(fields)
    spectra = [fields[row] + fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]
    assert len(spectra) == n_spec

    # Prepare config dictionary
    config = {
        'obs_bp': obs_bp,
        'inv_cov': inv_cov,
        'mix_lmin': mix_lmin,
        'mix_lmax': mix_lmax,
        'pos_nl': pos_nl,
        'she_nl': she_nl,
        'input_lmax': input_lmax,
        'n_spec': n_spec,
        'n_cl': n_cl,
        'n_zbin': n_zbin,
        'spectra': spectra,
        'n_bandpower': n_bandpower,
        'mixmat_nn_to_nn': mixmat_nn_to_nn,
        'mixmat_ne_to_ne': mixmat_ne_to_ne,
        'mixmat_ee_to_ee': mixmat_ee_to_ee,
        'mixmat_bb_to_ee': mixmat_bb_to_ee,
        'binmix_nn_to_nn': binmix_nn_to_nn,
        'binmix_ne_to_ne': binmix_ne_to_ne,
        'binmix_ee_to_ee': binmix_ee_to_ee,
        'binmix_bb_to_ee': binmix_bb_to_ee
    }
    return config

'''

def setup(obs_bp_path, binmixmat_path, mix_lmin, cov_path, pos_nl_path, she_nl_path, noise_lmin, input_lmax,
          n_zbin):
    """
    Load and precompute everything that is fixed throughout parameter space. This should be called once per analysis,
    prior to any calls to execute.

    Args:
        obs_bp_path (str): Path to the observed bandpowers, in a numpy .npz file with array name obs_bp and shape
                           (n_spectra, n_bandpowers), with spectra in diagonal-major order.
        binmixmat_path (str): Path to combined mixing and binning matrices, in numpy .npz file with array names
                              (binmix_tt_to_tt, binmix_te_to_te, binmix_ee_to_ee, binmix_bb_to_ee), each with shape
                              (n_bandpower, input_lmax - mix_lmin + 1).
        mix_lmin (int): Minimum l for the theory power in the mixing matrices.
        cov_path (str): Path to precomputed covariance, in numpy .npz file with array name cov, with shape
                        (n_data, n_data) where n_data = n_spectra * n_bandpowers.
        pos_nl_path (str): Path to the unbinned position noise power spectrum, in text file.
        she_nl_path (str): Path to the unbinned shear noise power spectrum, in text file.
        noise_lmin (int): Minimum l in noise power spectra.
        input_lmax (int): Maximum l to include in mixing. Theory and noise power will be truncated above this.
        n_zbin (int): Number of redshift bins. It will be assumed that there is one position field and one shear field
                      per redshift bin.

    Returns:
        dict: Config dictionary to pass to execute.
    """

    # Load observed bandpowers and prepare into a vector
    with np.load(obs_bp_path) as data:
        obs_bp = data['obs_bp']
    n_spec, n_bandpower = obs_bp.shape
    assert n_spec == (2 * n_zbin) * (2 * n_zbin + 1) // 2
    n_data = n_spec * n_bandpower
    obs_bp = np.reshape(obs_bp, n_data)

    # Load covariance
    with np.load(cov_path) as data:
        cov = data['cov']
    assert cov.shape == (n_data, n_data)

    # Invert covariance
    inv_cov = np.linalg.inv(cov)

    # Load combined mixing and binning matrices
    with np.load(binmixmat_path) as data:
        binmix_nn_to_nn = data['binmix_tt_to_tt']
        binmix_ne_to_ne = data['binmix_te_to_te']
        binmix_ee_to_ee = data['binmix_ee_to_ee']
        binmix_bb_to_ee = data['binmix_bb_to_ee']
    n_cl = binmix_nn_to_nn.shape[1]
    assert binmix_nn_to_nn.shape == (n_bandpower, n_cl)
    assert binmix_ne_to_ne.shape == (n_bandpower, n_cl)
    assert binmix_ee_to_ee.shape == (n_bandpower, n_cl)
    assert binmix_bb_to_ee.shape == (n_bandpower, n_cl)
    mix_lmax = mix_lmin + n_cl - 1

    # Load noise and trim/pad to correct length for input to mixing matrices,
    # truncating power above input_lmax

    zeros_lowl = np.zeros(noise_lmin)
    zeros_hil = np.zeros(max(mix_lmax - input_lmax, 0))

    if pos_nl_path is not None:
        assert she_nl_path is not None
        pos_nl = np.loadtxt(pos_nl_path, max_rows=(input_lmax - noise_lmin + 1))
        she_nl = np.loadtxt(she_nl_path, max_rows=(input_lmax - noise_lmin + 1))

        pos_nl = np.concatenate((zeros_lowl, pos_nl, zeros_hil))[mix_lmin:(mix_lmax + 1)]
        she_nl = np.concatenate((zeros_lowl, she_nl, zeros_hil))[mix_lmin:(mix_lmax + 1)]
        assert pos_nl.shape == (n_cl, )
        assert she_nl.shape == (n_cl, )
    else:
        assert pos_nl_path is None
        assert she_nl_path is None
        pos_nl = None
        she_nl = None
    # Generate a list of spectrum types (NN, EE or NE) in the correct (diagonal) order, so that we know which mixing
    # matrix/matrices to apply
    fields = [field for _ in range(n_zbin) for field in ('N', 'E')]
    n_field = len(fields)
    spectra = [fields[row] + fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]
    assert len(spectra) == n_spec

    # Prepare config dictionary
    config = {
        'obs_bp': obs_bp,
        'inv_cov': inv_cov,
        'mix_lmin': mix_lmin,
        'mix_lmax': mix_lmax,
        'pos_nl': pos_nl,
        'she_nl': she_nl,
        'input_lmax': input_lmax,
        'n_spec': n_spec,
        'n_cl': n_cl,
        'n_zbin': n_zbin,
        'spectra': spectra,
        'n_bandpower': n_bandpower,
        'binmix_nn_to_nn': binmix_nn_to_nn,
        'binmix_ne_to_ne': binmix_ne_to_ne,
        'binmix_ee_to_ee': binmix_ee_to_ee,
        'binmix_bb_to_ee': binmix_bb_to_ee
    }
    return config

'''

def execute(theory_cl, theory_lmin, config, noise_cls, pbl, hpx_win=False, nside=64):
    """
    Calculate the joint log-likelihood at a particular point in parameter space.

    Args:
        theory_cl (2D numpy array): Theory power spectra, in diagonal ordering, with shape (n_spectra, n_ell).
        theory_lmin (int): Minimum l used in theory_cl.
        config (dict): Config dictionary returned by setup.

    Returns:
        float: Log-likelihood value.
    """

    # Unpack config dictionary
    obs_bp = config['obs_bp']
    inv_cov = config['inv_cov']
    mix_lmin = config['mix_lmin']
    mix_lmax = config['mix_lmax']
    pos_nl = config['pos_nl']
    she_nl = config['she_nl']
    input_lmax = config['input_lmax']
    n_spec = config['n_spec']
    n_cl = config['n_cl']
    n_zbin = config['n_zbin']
    spectra = config['spectra']
    n_bandpower = config['n_bandpower']
    mixmat_nn_to_nn = config['mixmat_nn_to_nn']
    mixmat_ne_to_ne = config['mixmat_ne_to_ne']
    mixmat_ee_to_ee = config['mixmat_ee_to_ee']
    mixmat_bb_to_ee = config['mixmat_bb_to_ee']
    binmix_nn_to_nn = config['binmix_nn_to_nn']
    binmix_ne_to_ne = config['binmix_ne_to_ne']
    binmix_ee_to_ee = config['binmix_ee_to_ee']
    binmix_bb_to_ee = config['binmix_bb_to_ee']

    # Trim/pad theory Cls to correct length for input to mixing matrices, truncating power above input_lmax:
    # 1. Trim so power is truncated above input_lmax
    theory_cl = theory_cl[:, :(input_lmax - theory_lmin + 1)]
    # 2. Pad so theory power runs from 0 up to max(input_lmax, mix_lmax)
    zeros_lowl = np.zeros((n_spec, theory_lmin))
    zeros_hil = np.zeros((n_spec, max(mix_lmax - input_lmax, 0)))
    theory_cl = np.concatenate((zeros_lowl, theory_cl, zeros_hil), axis=-1)
    # 3. Truncate so it runs from mix_lmin to mix_lmax
    theory_cl = theory_cl[:, mix_lmin:(mix_lmax + 1)]
    assert theory_cl.shape == (n_spec, n_cl), (theory_cl.shape, (n_spec, n_cl))

    #Now trim/pad noise Cls as above
    # 1. Trim so power is truncated above input_lmax
    noise_cls = noise_cls[:, :(input_lmax - theory_lmin + 1)]
    # 2. Pad so theory power runs from 0 up to max(input_lmax, mix_lmax)
    zeros_lowl = np.zeros((n_spec, theory_lmin))
    zeros_hil = np.zeros((n_spec, max(mix_lmax - input_lmax, 0)))
    noise_cls = np.concatenate((zeros_lowl, noise_cls, zeros_hil), axis=-1)
    # 3. Truncate so it runs from mix_lmin to mix_lmax
    noise_cls = noise_cls[:, mix_lmin:(mix_lmax + 1)]
    #assert noise_cls.shape == (n_spec, n_cl), (noise_cls.shape, (n_spec, n_cl))

    if noise_cls is None:
        # Add noise to auto-spectra
        theory_cl[:(2 * n_zbin):2] += pos_nl
        theory_cl[1:(2 * n_zbin):2] += she_nl
    #else:
    #    theory_cl = np.asarray(theory_cl) + np.asarray(noise_cls)

    exp_bp = np.full((n_spec, n_bandpower), np.nan)
    for spec_idx, spec in enumerate(spectra):

        if spec == 'NN':
            this_cl = theory_cl[spec_idx]
            this_noise_cl = noise_cls[spec_idx]
            hpx_win_correction = (hp.pixwin(nside=nside,pol=True,lmax=input_lmax)[0][mix_lmin:])**2
            if hpx_win is True:
                #this_exp_bp = pbl@((hpx_win_correction*(mixmat_nn_to_nn@this_cl))+(hpx_win_correction*this_noise_cl))
                #this_exp_bp = pbl@((hpx_win_correction*(mixmat_nn_to_nn@this_cl))+(this_noise_cl))
                this_exp_bp = pbl@((mixmat_nn_to_nn@(hpx_win_correction*this_cl))+(this_noise_cl))

            else:
                #print(this_noise_cl)
                this_exp_bp = pbl@((mixmat_nn_to_nn@(this_cl))+(this_noise_cl))
                #print(this_exp_bp)
                #print(mixmat_nn_to_nn)
                #print(mixmat_nn_to_nn@(this_cl))
        elif spec in ('NE', 'EN'):
            this_cl = theory_cl[spec_idx]
            this_noise_cl = noise_cls[spec_idx]
            hpx_win_correction = (hp.pixwin(nside=nside,pol=True,lmax=input_lmax)[0][mix_lmin:])*\
                                 (hp.pixwin(nside=nside,pol=True,lmax=input_lmax)[1][mix_lmin:])

            if hpx_win is True:
                #this_exp_bp = pbl@((hpx_win_correction*(mixmat_ne_to_ne@this_cl))+(this_noise_cl))
                this_exp_bp = pbl@((mixmat_ne_to_ne@(hpx_win_correction*this_cl))+this_noise_cl)
            else:
                this_exp_bp = pbl@((mixmat_ne_to_ne@this_cl)+this_noise_cl)

        elif spec == 'EE':
            this_cl = theory_cl[spec_idx]
            this_noise_cl = noise_cls[spec_idx]
            hpx_win_correction = (hp.pixwin(nside=nside,pol=True,lmax=input_lmax)[1][mix_lmin:])**2

            if hpx_win is True:
                if spec_idx < 2 * n_zbin:
                    #this_exp_bp = pbl@((hpx_win_correction*((mixmat_ee_to_ee@this_cl)+(mixmat_bb_to_ee@this_cl)))+this_noise_cl) #Add BB noise contribution to auto-spectra
                    #this_exp_bp = pbl@((hpx_win_correction*(mixmat_ee_to_ee@this_cl))+((this_noise_cl)))
                    this_exp_bp = pbl@((mixmat_ee_to_ee@(hpx_win_correction*this_cl))+(this_noise_cl))
                else:
                    #this_exp_bp = pbl@((hpx_win_correction*(mixmat_ee_to_ee@this_cl))+(this_noise_cl))
                    this_exp_bp = pbl@((mixmat_ee_to_ee@(hpx_win_correction*this_cl))+(this_noise_cl))

            else:
                if spec_idx < 2 * n_zbin:
                    this_exp_bp = pbl@((mixmat_ee_to_ee@this_cl)+this_noise_cl) #Add BB noise contribution to auto-spectra
                else:
                    this_exp_bp = pbl@((mixmat_ee_to_ee@this_cl)+this_noise_cl)
        else:
            raise ValueError('Unexpected spectrum: ' + spec)
        exp_bp[spec_idx] = this_exp_bp
    assert np.all(np.isfinite(exp_bp))

    # Vectorise
    exp_bp = np.reshape(exp_bp, n_spec * n_bandpower)
    #print(exp_bp)
    # Evalute log pdf
    return mvg_logpdf_fixedcov(obs_bp, exp_bp, inv_cov)

def exp_bp(theory_cl, theory_lmin, config, noise_cls, pbl, hpx_win=False, nside=64):
    """
    Calculate the joint log-likelihood at a particular point in parameter space.

    Args:
        theory_cl (2D numpy array): Theory power spectra, in diagonal ordering, with shape (n_spectra, n_ell).
        theory_lmin (int): Minimum l used in theory_cl.
        config (dict): Config dictionary returned by setup.

    Returns:
        float: Log-likelihood value.
    """

    # Unpack config dictionary
    obs_bp = config['obs_bp']
    inv_cov = config['inv_cov']
    mix_lmin = config['mix_lmin']
    mix_lmax = config['mix_lmax']
    pos_nl = config['pos_nl']
    she_nl = config['she_nl']
    input_lmax = config['input_lmax']
    n_spec = config['n_spec']
    n_cl = config['n_cl']
    n_zbin = config['n_zbin']
    spectra = config['spectra']
    n_bandpower = config['n_bandpower']
    mixmat_nn_to_nn = config['mixmat_nn_to_nn']
    mixmat_ne_to_ne = config['mixmat_ne_to_ne']
    mixmat_ee_to_ee = config['mixmat_ee_to_ee']
    mixmat_bb_to_ee = config['mixmat_bb_to_ee']
    binmix_nn_to_nn = config['binmix_nn_to_nn']
    binmix_ne_to_ne = config['binmix_ne_to_ne']
    binmix_ee_to_ee = config['binmix_ee_to_ee']
    binmix_bb_to_ee = config['binmix_bb_to_ee']

    # Trim/pad theory Cls to correct length for input to mixing matrices, truncating power above input_lmax:
    # 1. Trim so power is truncated above input_lmax
    theory_cl = theory_cl[:, :(input_lmax - theory_lmin + 1)]
    # 2. Pad so theory power runs from 0 up to max(input_lmax, mix_lmax)
    zeros_lowl = np.zeros((n_spec, theory_lmin))
    zeros_hil = np.zeros((n_spec, max(mix_lmax - input_lmax, 0)))
    theory_cl = np.concatenate((zeros_lowl, theory_cl, zeros_hil), axis=-1)
    # 3. Truncate so it runs from mix_lmin to mix_lmax
    theory_cl = theory_cl[:, mix_lmin:(mix_lmax + 1)]
    assert theory_cl.shape == (n_spec, n_cl), (theory_cl.shape, (n_spec, n_cl))

    #Need to adjust theory Cl for Healpix window function if measuring from maps
    #hpx_window = hp.pixwin(nside=64, pol=True, lmax=input_lmax)

    if noise_cls is None:
        # Add noise to auto-spectra
        theory_cl[:(2 * n_zbin):2] += pos_nl
        theory_cl[1:(2 * n_zbin):2] += she_nl
    #else:
    #    theory_cl = np.asarray(theory_cl) + np.asarray(noise_cls)

    exp_bp = np.full((n_spec, n_bandpower), np.nan)

    for spec_idx, spec in enumerate(spectra):

        if spec == 'NN':
            this_cl = theory_cl[spec_idx]
            this_noise_cl = noise_cls[spec_idx]
            hpx_win_correction = (hp.pixwin(nside=nside,pol=True,lmax=input_lmax)[0][mix_lmin:])**2
            if hpx_win is True:
                #this_exp_bp = (binmix_nn_to_nn @ (this_cl*(hp.pixwin(nside=64,pol=True,lmax=input_lmax)[0][mix_lmin:])**2))+this_noise_cl
                #this_exp_bp = binmix_nn_to_nn @ (theory_cl[spec_idx]*(hp.pixwin(nside=64,pol=True,lmax=input_lmax)[0][mix_lmin:])**2)
                #this_exp_bp = binmix_nn_to_nn @ (this_cl*(hp.pixwin(nside=64,pol=True,lmax=input_lmax)[0][mix_lmin:])**2) + pbl@this_noise_cl
                #this_exp_bp = pbl@((hpx_win_correction*(mixmat_nn_to_nn@this_cl))+(hpx_win_correction*this_noise_cl))
                #this_exp_bp = pbl@((hpx_win_correction*(mixmat_nn_to_nn@this_cl))+(this_noise_cl))
                this_exp_bp = pbl@((mixmat_nn_to_nn@(hpx_win_correction*this_cl))+(this_noise_cl))
            else:
                #this_exp_bp = (binmix_nn_to_nn @ this_cl)+this_noise_cl
                #this_exp_bp = binmix_nn_to_nn @ theory_cl[spec_idx] + pbl@this_noise_cl
                this_exp_bp = pbl@((mixmat_nn_to_nn@(this_cl))+(this_noise_cl))

        elif spec in ('NE', 'EN'):
            this_cl = theory_cl[spec_idx]
            this_noise_cl = noise_cls[spec_idx]
            hpx_win_correction = (hp.pixwin(nside=nside,pol=True,lmax=input_lmax)[0][mix_lmin:])*\
                                 (hp.pixwin(nside=nside,pol=True,lmax=input_lmax)[1][mix_lmin:])

            if hpx_win is True:
                #this_exp_bp = (binmix_ne_to_ne @
                #               (this_cl*
                #                (hp.pixwin(nside=64,pol=True,lmax=input_lmax)[0][mix_lmin:])*
                #                hp.pixwin(nside=64,pol=True,lmax=input_lmax)[1][mix_lmin:])) + this_noise_cl

                #this_exp_bp = binmix_ne_to_ne @ (theory_cl[spec_idx]*
                #                                 hp.pixwin(nside=64,pol=True,lmax=input_lmax)[0][mix_lmin:]*
                #                                 hp.pixwin(nside=64,pol=True,lmax=input_lmax)[1][mix_lmin:])

                #this_exp_bp = binmix_ne_to_ne@(this_cl*hp.pixwin(nside=64,pol=True,lmax=input_lmax)[0][mix_lmin:]*hp.pixwin(nside=64,pol=True,lmax=input_lmax)[1][mix_lmin:]) + pbl@this_noise_cl
                #this_exp_bp = pbl@((hpx_win_correction*(mixmat_ne_to_ne@this_cl))+this_noise_cl)
                this_exp_bp = pbl@((mixmat_ne_to_ne@(hpx_win_correction*this_cl))+this_noise_cl)
            else:
                #this_exp_bp  =(binmix_ne_to_ne@this_cl) + this_noise_cl
                #this_exp_bp = binmix_ne_to_ne @ this_cl + pbl@this_noise_cl
                this_exp_bp = pbl@((mixmat_ne_to_ne@this_cl)+this_noise_cl)

        elif spec == 'EE':
            this_cl = theory_cl[spec_idx]
            this_noise_cl = noise_cls[spec_idx]
            hpx_win_correction = (hp.pixwin(nside=nside,pol=True,lmax=input_lmax)[1][mix_lmin:])**2

            if hpx_win is True:
                #this_exp_bp = (binmix_ee_to_ee @ (this_cl*(hp.pixwin(nside=64,pol=True,lmax=input_lmax)[1][mix_lmin:])**2))+this_noise_cl
                #this_exp_bp = binmix_ee_to_ee @ (theory_cl[spec_idx]*(hp.pixwin(nside=64,pol=True,lmax=input_lmax)[1][mix_lmin:])**2)
                #this_exp_bp = binmix_ee_to_ee @ (this_cl*(hp.pixwin(nside=64,pol=True,lmax=input_lmax)[1][mix_lmin:])**2) + pbl@this_noise_cl
                if spec_idx < 2 * n_zbin:
                    #this_exp_bp = pbl@((hpx_win_correction*((mixmat_ee_to_ee@this_cl)+(mixmat_bb_to_ee@this_cl)))+this_noise_cl) #Add BB noise contribution to auto-spectra
                    #this_exp_bp = pbl@((hpx_win_correction*(mixmat_ee_to_ee@this_cl))+(this_noise_cl))
                    this_exp_bp = pbl@((mixmat_ee_to_ee@(hpx_win_correction*this_cl))+(this_noise_cl))

                else:
                    #this_exp_bp = pbl@((hpx_win_correction*(mixmat_ee_to_ee@this_cl))+this_noise_cl)
                    this_exp_bp = pbl@((mixmat_ee_to_ee@(hpx_win_correction*this_cl))+(this_noise_cl))

            else:
                #this_exp_bp = (binmix_ee_to_ee @ this_cl) + this_noise_cl
                #this_exp_bp = binmix_ee_to_ee @ this_cl + pbl@this_noise_cl
                #this_exp_bp = pbl@((mixmat_ee_to_ee@this_cl)+this_noise_cl)
                if spec_idx < 2 * n_zbin:
                    #this_exp_bp = pbl@(((mixmat_ee_to_ee@this_cl)+(mixmat_bb_to_ee@this_cl))+this_noise_cl) #Add BB noise contribution to auto-spectra
                    this_exp_bp = pbl@((mixmat_ee_to_ee@this_cl)+this_noise_cl) #Add BB noise contribution to auto-spectra
                else:
                    this_exp_bp = pbl@((mixmat_ee_to_ee@this_cl)+this_noise_cl)
        else:
            raise ValueError('Unexpected spectrum: ' + spec)
        exp_bp[spec_idx] = this_exp_bp
    assert np.all(np.isfinite(exp_bp))
    #exp_bp = np.reshape(exp_bp, n_spec * n_bandpower)

    #print(exp_bp)
    return exp_bp


def execute_old(theory_cl, theory_lmin, config, noise_cls=None, hpx_win=False):
    """
    Calculate the joint log-likelihood at a particular point in parameter space.

    Args:
        theory_cl (2D numpy array): Theory power spectra, in diagonal ordering, with shape (n_spectra, n_ell).
        theory_lmin (int): Minimum l used in theory_cl.
        config (dict): Config dictionary returned by setup.

    Returns:
        float: Log-likelihood value.
    """

    # Unpack config dictionary
    obs_bp = config['obs_bp']
    inv_cov = config['inv_cov']
    mix_lmin = config['mix_lmin']
    mix_lmax = config['mix_lmax']
    pos_nl = config['pos_nl']
    she_nl = config['she_nl']
    input_lmax = config['input_lmax']
    n_spec = config['n_spec']
    n_cl = config['n_cl']
    n_zbin = config['n_zbin']
    spectra = config['spectra']
    n_bandpower = config['n_bandpower']
    binmix_nn_to_nn = config['binmix_nn_to_nn']
    binmix_ne_to_ne = config['binmix_ne_to_ne']
    binmix_ee_to_ee = config['binmix_ee_to_ee']
    binmix_bb_to_ee = config['binmix_bb_to_ee']

    # Trim/pad theory Cls to correct length for input to mixing matrices, truncating power above input_lmax:
    # 1. Trim so power is truncated above input_lmax
    theory_cl = theory_cl[:, :(input_lmax - theory_lmin + 1)]
    # 2. Pad so theory power runs from 0 up to max(input_lmax, mix_lmax)
    zeros_lowl = np.zeros((n_spec, theory_lmin))
    zeros_hil = np.zeros((n_spec, max(mix_lmax - input_lmax, 0)))
    theory_cl = np.concatenate((zeros_lowl, theory_cl, zeros_hil), axis=-1)
    # 3. Truncate so it runs from mix_lmin to mix_lmax
    theory_cl = theory_cl[:, mix_lmin:(mix_lmax + 1)]
    assert theory_cl.shape == (n_spec, n_cl), (theory_cl.shape, (n_spec, n_cl))

    #Need to adjust theory Cl for Healpix window function if measuring from maps
    #hpx_window = hp.pixwin(nside=64, pol=True, lmax=input_lmax)

    if noise_cls is None:
        # Add noise to auto-spectra
        theory_cl[:(2 * n_zbin):2] += pos_nl
        theory_cl[1:(2 * n_zbin):2] += she_nl
    else:
        theory_cl = np.asarray(theory_cl) + np.asarray(noise_cls)

    exp_bp = np.full((n_spec, n_bandpower), np.nan)
    for spec_idx, spec in enumerate(spectra):
        if spec == 'NN':
            if hpx_win is True:
                this_exp_bp = binmix_nn_to_nn @ (theory_cl[spec_idx]*(hp.pixwin(nside=64,pol=True,lmax=input_lmax)[0][mix_lmin:])**2)
            else:
                this_exp_bp = binmix_nn_to_nn @ theory_cl[spec_idx]
                #this_exp_bp += noise_cls_trimmed[spec_idx]
        elif spec in ('NE', 'EN'):
            if hpx_win is True:
                this_exp_bp = binmix_ne_to_ne @ (theory_cl[spec_idx]*
                                                 hp.pixwin(nside=64,pol=True,lmax=input_lmax)[0][mix_lmin:]*
                                                 hp.pixwin(nside=64,pol=True,lmax=input_lmax)[1][mix_lmin:])
            else:
                this_exp_bp = binmix_ne_to_ne @ theory_cl[spec_idx]
                #this_exp_bp += noise_cls_trimmed[spec_idx]
        elif spec == 'EE':
            if hpx_win is True:
                this_exp_bp = binmix_ee_to_ee @ (theory_cl[spec_idx]*(hp.pixwin(nside=64,pol=True,lmax=input_lmax)[1][mix_lmin:])**2)
            else:
                this_exp_bp = binmix_ee_to_ee @ theory_cl[spec_idx]
                #this_exp_bp += noise_cls_trimmed[spec_idx]

            if spec_idx < 2 * n_zbin:
                #Check this!!
                #print('BB NOISE BELOW:')
                #print(noise_cls[spec_idx])
                this_exp_bp += binmix_bb_to_ee @ noise_cls[spec_idx] # BB noise contribution to auto-spectra
        else:
            raise ValueError('Unexpected spectrum: ' + spec)
        exp_bp[spec_idx] = this_exp_bp
    assert np.all(np.isfinite(exp_bp))

    # Apply Hpx window function
    '''
    # Apply mixing/binning matrices to obtain bandpower expectation
    exp_bp = np.full((n_spec, n_bandpower), np.nan)
    for spec_idx, spec in enumerate(spectra):
        if spec == 'NN':
            this_exp_bp = binmix_nn_to_nn @ theory_cl[spec_idx]
            #this_exp_bp += noise_cls_trimmed[spec_idx]
        elif spec in ('NE', 'EN'):
            this_exp_bp = binmix_ne_to_ne @ theory_cl[spec_idx]
            #this_exp_bp += noise_cls_trimmed[spec_idx]
        elif spec == 'EE':
            this_exp_bp = binmix_ee_to_ee @ theory_cl[spec_idx]
            #this_exp_bp += noise_cls_trimmed[spec_idx]

            if spec_idx < 2 * n_zbin:
                #Check this!!
                #print('BB NOISE BELOW:')
                #print(noise_cls[spec_idx])
                this_exp_bp += binmix_bb_to_ee @ noise_cls[spec_idx] # BB noise contribution to auto-spectra
        else:
            raise ValueError('Unexpected spectrum: ' + spec)
        exp_bp[spec_idx] = this_exp_bp
    assert np.all(np.isfinite(exp_bp))
    '''


    '''
    if noise_cls is None:
        # Add noise to auto-spectra
        theory_cl[:(2 * n_zbin):2] += pos_nl
        theory_cl[1:(2 * n_zbin):2] += she_nl
    else:
        theory_cl = np.asarray(theory_cl) + np.asarray(noise_cls)

    # Apply mixing/binning matrices to obtain bandpower expectation
    exp_bp = np.full((n_spec, n_bandpower), np.nan)
    for spec_idx, spec in enumerate(spectra):
        if spec == 'NN':
            this_exp_bp = binmix_nn_to_nn @ theory_cl[spec_idx]
        elif spec in ('NE', 'EN'):
            this_exp_bp = binmix_ne_to_ne @ theory_cl[spec_idx]
        elif spec == 'EE':
            this_exp_bp = binmix_ee_to_ee @ theory_cl[spec_idx]
            if spec_idx < 2 * n_zbin:
                #Check this!!
                #print('BB NOISE BELOW:')
                #print(noise_cls[spec_idx])
                this_exp_bp += binmix_bb_to_ee @ noise_cls[spec_idx] # BB noise contribution to auto-spectra
        else:
            raise ValueError('Unexpected spectrum: ' + spec)
        exp_bp[spec_idx] = this_exp_bp
    assert np.all(np.isfinite(exp_bp))
    print('BP Below:')
    print(exp_bp)
    '''

    #np.savetxt('/raid/scratch/wongj/mywork/3x2pt/TEST_3_BINS_NEW/inference_analysis/bps/bp.txt',np.transpose(exp_bp))
    # Vectorise
    exp_bp = np.reshape(exp_bp, n_spec * n_bandpower)
    #print(exp_bp)
    # Evalute log pdf
    return mvg_logpdf_fixedcov(obs_bp, exp_bp, inv_cov)
