"""
Likelihood module to evaluate the joint likelihood of a set of tomographic 3x2pt power spectra, binned into bandpowers,
on the full sky using a multivariate Gaussian likelihood.

The main functions are setup, which should be called once per analysis, and execute, which is called for every new
point in parameter space.
"""

import enum
import os.path

import numpy as np

class FieldType(enum.Enum):
    """
    Simple class to represent the three types of fields: position, shear E and shear B.
    """

    POS = enum.auto()
    SHE_E = enum.auto()
    SHE_B = enum.auto()


def idx_to_type(idx):
    """
    Returns the field type corresponding to a row or column index in the matrix of power spectra. Shear B-mode is not
    taken into account.

    Args:
        idx (int): Row or column index.

    Returns:
        FieldType: The corresponding field type.
    """

    remainder = idx % 2
    return FieldType.POS if remainder == 0 else FieldType.SHE_E


def idx_to_zbin(idx):
    """
    Returns the redshift bin corresponding to a row or column index in the matrix of power spectra. Redshift bins are
    numbered from 1, and shear B-mode is not taken into account.

    Args:
        idx (int): Row or column index.

    Returns:
        int: The corresponding redshift bin number.
    """

    return 1 + idx // 2


def load_spectra(n_zbin, pos_pos_dir, she_she_dir, pos_she_dir, lmax=None, lmin=0):
    """
    Given the number of redshift bins and relevant directories, load (band)power spectra (position, shear, cross) in the
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

    # Load (band)power spectra in diagonal-major order
    spectra = []
    for diag in range(n_field):
        for row in range(n_field - diag):
            col = row + diag

            # Determine the field types for this row and column
            row_type = idx_to_type(row)
            col_type = idx_to_type(col)

            # Determine the redshift bins for this row and column, and order them to match cosmosis output:
            # for pos-pos and she-she the higher bin index goes first, for pos-she pos goes first
            bins = (idx_to_zbin(row), idx_to_zbin(col))
            if row_type == col_type: # pos-pos or she-she
                bin1 = max(bins)
                bin2 = min(bins)
            elif row_type == FieldType.POS: # pos-she
                bin1, bin2 = bins
            else: # she-pos, so invert
                bin2, bin1 = bins

            # Determine the input path
            if row_type == FieldType.POS and col_type == FieldType.POS:
                cl_dir = pos_pos_dir
            elif row_type == FieldType.SHE_E and col_type == FieldType.SHE_E:
                cl_dir = she_she_dir
            else:
                cl_dir = pos_she_dir
            filename = f'bin_{bin1}_{bin2}.txt'
            cl_path = os.path.join(cl_dir, filename)
            # Load with appropriate ell_range
            max_rows = None if lmax is None else (lmax - lmin + 1)

            spec = np.concatenate((np.zeros(lmin), np.loadtxt(cl_path, max_rows=max_rows).reshape(-1)))

            if spec.ndim < 1:
                spec = spec[np.newaxis]
            spectra.append(spec)

    return np.asarray(spectra)

def load_spectra_1x2pt(n_zbin, she_she_dir, lmax=None, lmin=0):
    """
    Given the number of redshift bins and relevant directories, load (band)power spectra (shear) in the
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

    # Load (band)power spectra in diagonal-major order
    spectra = []
    for diag in range(n_field):
        for row in range(n_field - diag):
            col = row + diag

            # Determine the field types for this row and column
            row_type = idx_to_type(row)
            col_type = idx_to_type(col)

            # Determine the redshift bins for this row and column, and order them to match cosmosis output:
            # for pos-pos and she-she the higher bin index goes first, for pos-she pos goes first
            bins = (idx_to_zbin(row), idx_to_zbin(col))
            if row_type == col_type: # pos-pos or she-she
                bin1 = max(bins)
                bin2 = min(bins)
            elif row_type == FieldType.POS: # pos-she
                bin1, bin2 = bins
            else: # she-pos, so invert
                bin2, bin1 = bins

            # Determine the input path
            if row_type == FieldType.SHE_E and col_type == FieldType.SHE_E:
                cl_dir = she_she_dir
                filename = f'bin_{bin1}_{bin2}.txt'
                cl_path = os.path.join(cl_dir, filename)
                # Load with appropriate ell_range
                max_rows = None if lmax is None else (lmax - lmin + 1)

                spec = np.concatenate((np.zeros(lmin), np.loadtxt(cl_path, max_rows=max_rows).reshape(-1)))

                if spec.ndim < 1:
                    spec = spec[np.newaxis]
                spectra.append(spec)
            else:
                pass

    return np.asarray(spectra)


def setup(n_zbin, obs_pos_pos_dir, obs_she_she_dir, obs_pos_she_dir, pos_nl_path, she_nl_path, noise_ell_path, pbl_path,
          inv_cov_path, lmax, lmin):
    """
    Load and precompute everything that is fixed throughout parameter space. This should be called once per analysis,
    prior to any calls to execute.

    Args:
        n_zbin (int): Number of redshift bins. It will be assumed that there is one position field and one shear field
                      per redshift bin.
        obs_pos_pos_dir (str): Path to the directory containing the observed position-position band power spectra.
        obs_she_she_dir (str): Path to the directory containing the observed shear-shear band power spectra.
        obs_pos_she_dir (str): Path to the directory containing the observed position-shear band power spectra.
        pos_nl_path (str): Path to the unbinned position noise power spectrum, in text file.
        she_nl_path (str): Path to the unbinned shear noise power spectrum, in text file.
        noise_ell_path (str): Path to the text file containing the ells for the noise power spectra.
        pbl_path (str): Path to binning matrix, in text file with shape (n_bandpowers, lmax - lmin + 1).
        inv_cov_path (str): Path to precomputed binned inverse covariance, in numpy .npz file with array name inv_cov,
                            and shape (n_spectra * n_bandpowers, n_spectra * n_bandpowers).
        lmax (int): Maximum l to use in the likelihood.
        lmin (int): Minimum l.

    Returns:
        dict: Config dictionary to pass to execute.
    """

    # Calculate number of fields assuming 2 per redshift bin (shear B-modes not included in likelihood)
    n_fields = 2 * n_zbin

    # Load observed bandpowers and Pbl matrix
    obs_bp = load_spectra(n_zbin, obs_pos_pos_dir, obs_she_she_dir, obs_pos_she_dir)
    # JW added below!

    pbl = np.loadtxt(pbl_path)
    if pbl.ndim == 1 and lmax > lmin:
        pbl = pbl[np.newaxis, ...]

    # Do some consistency checks
    n_bp, n_ell = pbl.shape
    n_spectra = n_fields * (n_fields + 1) // 2

    assert n_ell == lmax - lmin + 1
    assert obs_bp.shape == (n_spectra, n_bp)

    # Flatten the observed bandpowers so it is suitable as input to a multivariate Gaussian
    obs_bp = obs_bp.flatten()

    # Load noise Cls & ells
    pos_nl = np.loadtxt(pos_nl_path)
    she_nl = np.loadtxt(she_nl_path)
    noise_ell = np.loadtxt(noise_ell_path)

    # Trim the ell range
    ell = np.arange(lmin, lmax + 1)
    pos_nl = pos_nl[(noise_ell >= lmin) & (noise_ell <= lmax)]
    she_nl = she_nl[(noise_ell >= lmin) & (noise_ell <= lmax)]
    assert len(pos_nl) == n_ell
    assert len(she_nl) == n_ell

    # Form noise Cl array, where noise is only added to auto-spectra
    nl_nonzero = np.array([pos_nl, she_nl]*n_zbin)
    nl_zero = np.zeros((n_spectra - n_fields, n_ell))
    nl = np.concatenate((nl_nonzero, nl_zero))

    # Load inverse covariance matrix of bandpowers
    with np.load(inv_cov_path) as data:
        inv_cov = data['inv_cov']
    assert inv_cov.shape == (n_spectra * n_bp, n_spectra * n_bp)

    # Form config dictionary
    config = {
        'ell': ell,
        'lmin': lmin,
        'lmax': lmax,
        'obs_bp': obs_bp,
        'noise_cl': nl,
        'pbl': pbl,
        'inv_cov': inv_cov
    }

    return config


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

    dev = x - mean
    return -0.5 * (dev @ inv_cov @ dev)


def joint_log_likelihood(obs_bp, theory_cl, noise_cl, pbl, inv_cov):
    """
    Returns the joint log-likelihood of all bandpowers for all spectra.

    Args:
        obs_bp (1D numpy array): Observed bandpowers, flattened into a 1D array ordered by spectrum then bandpower.
        theory_cl (2D numpy array): Theory power spectra, with shape (n_spectra, n_ell).
        noise_cl (2D numpy array): Noise power spectra in the same shape as theory_cl.
        pbl (2D numpy array): Bandpower binning matrix, with shape (n_bandpowers, n_ell).
        inv_cov (2D numpy array): Inverted covariance matrix, with shape
                                  (n_spectra * n_bandpowers, n_spectra * n_bandpowers).

    Returns:
        float: Joint log-likelihood value.
    """

    # Obtain mean by adding theory and noise Cls, then binning into bandpowers
    theory_cl += noise_cl
    theory_bp = np.einsum('bl,sl->sb', pbl, theory_cl)
    mean = np.ravel(theory_bp)

    return mvg_logpdf_fixedcov(obs_bp, mean, inv_cov)


def execute(theory_ell, theory_cl, config):
    """
    Perform some consistency checks then evaluate the likelihood for a given set of theory Cls.

    Args:
        theory_ell (1D numpy array): Ell range for all of the theory spectra (must be consistent between spectra).
        theory_cl (2D numpy array): Theory power spectra, in diagonal ordering, with shape (n_spectra, n_ell).
        config (dict): Config dictionary returned by setup.

    Returns:
        float: Log-likelihood value.
    """

    # Pull fixed (model-independent) parameters from config
    ell = config['ell']
    lmin = config['lmin']
    lmax = config['lmax']
    obs_bp = config['obs_bp']
    noise_cl = config['noise_cl']
    pbl = config['pbl']
    inv_cov = config['inv_cov']

    # Trim theory to ell range
    ell_to_keep = (theory_ell >= lmin) & (theory_ell <= lmax)
    assert np.allclose(ell, theory_ell[ell_to_keep])
    theory_cl = theory_cl[:, ell_to_keep]

    # Evaluate the likelihood
    return joint_log_likelihood(obs_bp, theory_cl, noise_cl, pbl, inv_cov)
