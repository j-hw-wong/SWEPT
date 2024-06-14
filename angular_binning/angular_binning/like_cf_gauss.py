"""
Likelihood module to evaluate the joint likelihood of a set of tomographic 3x2pt angular-binned correlation functions
using a multivariate Gaussian likelihood, where everything (observation, theory, covariance) is calculated
via power spectra.

The main functions are setup, which should be called once per analysis, and execute, which is called for every new
point in parameter space.
"""

import enum
import math
import os.path
from collections import namedtuple

import numpy as np
import scipy.special


DEG_TO_RAD = math.pi / 180.0


# Simple classes to help keep this understandable
Field = namedtuple('Field', ['field_type', 'zbin'])
PowerSpectrum = namedtuple('PowerSpectrum', ['field_1', 'field_2'])
class FieldType(enum.Enum):
    """
    The three types of fields: position, shear E and shear B.
    """
    POSITION = enum.auto()
    SHEAR_E = enum.auto()
    SHEAR_B = enum.auto()


def keep_spectrum(spec):
    """
    Returns False if the power spectrum is a type we wish to exclude (EB, NB), or True otherwise.

    Args:
        spec (PowerSpectrum): Power spectrum.

    Returns:
        bool: True if the power spectrum should be kept (i.e. is not EB or NB), or False otherwise.
    """
    field_types = (spec.field_1.field_type, spec.field_2.field_type)
    return not(FieldType.SHEAR_B in field_types
               and field_types != (FieldType.SHEAR_B, FieldType.SHEAR_B))


def idx_to_type(idx):
    """
    Returns the field type (position, shear E-mode or shear B-mode) corresponding to a row or column index in the matrix
    of power spectra, based on the remainder when dividing by 3.

    Args:
        idx (int): Row or column index.

    Returns:
        FieldType: The corresponding field type.
    """

    remainder = idx % 3
    return FieldType.POSITION if remainder == 0 else (FieldType.SHEAR_E if remainder == 1 else FieldType.SHEAR_B)


def idx_to_zbin(idx):
    """
    Returns the redshift bin corresponding to a row or column index in the matrix of power spectra, taking shear B-mode
    into account. Redshift bins are numbered from 1.

    Args:
        idx (int): Row or column index.

    Returns:
        int: The corresponding redshift bin number.
    """

    return 1 + idx // 3


def load_cls_zerob(n_zbin, pos_pos_dir, she_she_dir, pos_she_dir, lmax, lmin_in=0):
    """
    Given the number of redshift bins and relevant directories, load power spectra (position, shear, cross) in the
    correct order (diagonal / healpy new=True ordering) and insert B-mode power spectra all equal to zero.
    If lmin is supplied, the output will be padded to begin at l=0.

    Args:
        n_zbin (int): Number of redshift bins.
        pos_pos_dir (str): Path to directory containing position-position power spectra.
        she_she_dir (str): Path to directory containing shear-shear power spectra.
        pos_she_dir (str): Path to directory containing position-shear power spectra.
        lmax (int): Maximum l to load.
        lmin (int, optional): Minimum l supplied. Output will be padded with zeros below this point.

    Returns:
        (2D numpy array, list): First element of tuple is all Cls, with different spectra along the first axis and \
                                increasing l along the second. Second element of tuple is list of indices of spectra \
                                that involve shear B-modes, i.e. BB, EB, NB.
    """

    # Calculate number of fields, assuming 1 position field and 1 spin-2 shear field per redshift bin
    n_field = 3 * n_zbin

    # Load power spectra in diagonal-major order
    spectra = []
    b_indices = []
    i = -1
    for diag in range(n_field):
        for row in range(n_field - diag):
            col = row + diag
            i += 1

            # Determine the field types for this row and column
            row_type = idx_to_type(row)
            col_type = idx_to_type(col)

            # If either row or column is shear B-mode, the input spectrum is zero
            if row_type == FieldType.SHEAR_B or col_type == FieldType.SHEAR_B:
                spectra.append(np.zeros(lmax + 1))
                b_indices.append(i)
                continue

            # Determine the redshift bins for this row and column, and order them for cosmosis output:
            # for pos-pos and she-she the higher bin index goes first, for pos-she pos goes first
            bins = (idx_to_zbin(row), idx_to_zbin(col))
            if row_type == col_type: # pos-pos or she-she
                bin1 = max(bins)
                bin2 = min(bins)
            elif row_type == FieldType.POSITION: # pos-she
                bin1, bin2 = bins
            else: # she-pos, so invert
                bin2, bin1 = bins

            # Determine the input path
            if row_type == FieldType.POSITION and col_type == FieldType.POSITION:
                cl_dir = pos_pos_dir
            elif row_type == FieldType.SHEAR_E and col_type == FieldType.SHEAR_E:
                cl_dir = she_she_dir
            else:
                cl_dir = pos_she_dir
            cl_path = os.path.join(cl_dir, f'bin_{bin1}_{bin2}.txt')

            # Load with appropriate ell range
            spec = np.concatenate((np.zeros(lmin_in), np.loadtxt(cl_path, max_rows=(lmax - lmin_in + 1))))
            spectra.append(spec)

    assert len(spectra) - 1 == i
    return np.asarray(spectra), b_indices


def matrix_indices(vector_idx, matrix_size):
    """
    Return matrix indices (of Wishart scale matrix) corresponding to vector index (vector of Cls in diagonal-major
    order)

    Args:
        vector_idx (int): Vector index.
        matrix_size (int): Size along one axis of the square matrix to output indices for.

    Returns:
        (int, int): Row index, column index.
    """

    assert vector_idx < matrix_size * (matrix_size + 1) / 2, 'Invalid vector_idx for this matrix_size'

    # Work out which diagonal the element is on and its index on that diagonal, by iterating over the diagonals
    diag_length = matrix_size
    while vector_idx - diag_length >= 0:
        vector_idx -= diag_length
        diag_length -= 1
    diag = matrix_size - diag_length

    # Index at the top of the diagonal is (row = 0, col = diag),
    # so index of element is (row = vector_idx, col = diag + vector_idx)
    row = vector_idx
    col = diag + vector_idx
    return row, col


def vector_index(row_idx, col_idx, matrix_size):
    """
    Return vector index (vector of Cls in healpy diagonal-major order) corresponding to matrix indices (of Wishart
    scale matrix).

    Args:
        row_idx (int): Row index for matrix.
        col_idx (int): Column index for matrix.
        matrix_size (int): Size along one axis of the square matrix that input indices are for.

    Returns:
        int: Vector index corresponding to the input matrix indices.
    """

    # Only consider the upper triangle of the matrix by requiring that col >= row
    col = max(row_idx, col_idx)
    row = min(row_idx, col_idx)

    # Formula comes from standard sum over n
    diag = col - row
    return int(row + diag * (matrix_size - 0.5 * (diag - 1)))


def calculate_cl_cov_l(theory_cl, l, n_fields):
    """
    Returns the Gaussian covariance matrix of full-sky Cl estimates for a single l.

    Args:
        theory_cl (1D numpy array): All Cls for this l, in diagonal ordering.
        l (int): The l value.
        n_fields (int): The number of fields, where the number of spectra = n_fields * (n_fields + 1) / 2.

    Returns:
        2D numpy array: The covariance matrix for this l.
    """

    # Create empty covariance matrix
    n_spectra = len(theory_cl)
    cov_mat = np.full((n_spectra, n_spectra), np.nan)

    # Loop over all combinations of theory Cls
    for i, _ in enumerate(theory_cl):
        alpha, beta = matrix_indices(i, n_fields)

        for j, _ in enumerate(theory_cl):

            # Calculate all the relevant indices
            gamma, epsilon = matrix_indices(j, n_fields)
            alpha_gamma_idx = vector_index(alpha, gamma, n_fields)
            beta_epsilon_idx = vector_index(beta, epsilon, n_fields)
            alpha_epsilon_idx = vector_index(alpha, epsilon, n_fields)
            beta_gamma_idx = vector_index(beta, gamma, n_fields)

            # Calculate the covariance using the general Gaussian covariance equation
            # (see arXiv:2012.06267 eqn 6)
            cov = (theory_cl[alpha_gamma_idx] * theory_cl[beta_epsilon_idx]
                   + theory_cl[alpha_epsilon_idx] * theory_cl[beta_gamma_idx]) / (2 * l + 1.)
            cov_mat[i, j] = cov

    # Check for finite and symmetric - PD is not checked here because numerical issues mean that a valid set of Cls
    # can give a very slightly non-PD covariance matrix, but this doesn't affect the fixed-covariance results
    assert np.all(np.isfinite(cov_mat)), 'Covariance matrix not finite'
    assert np.allclose(cov_mat, cov_mat.T), 'Covariance matrix not symmetric'

    return cov_mat


def calculate_cl_covs(fid_cl, n_zbin, keep_spectra, lmin):
    """
    Calculate and return all full-sky power spectrum covariance matrices.

    Args:
        fid_cl (2D numpy array): Fiducial power spectra used to calculate the covariance, with shape (n_spectra, n_ell),
                                 with the spectra in diagonal-major order and the ells starting with l = lmin.
        n_zbin (int): Number of redshift bins. 3 fields per redshift bin are assumed: position, shear E, shear B.
        keep_spectra (list): Boolean list of whether each power spectrum should be kept: False if NB or EB,
                             otherwise True.
        lmin (int): Minimum l value.

    Returns:
        3D numpy array: Covariance with shape (n_ell, n_spectra, n_spectra).
    """

    # Calculate Cl covariance for each l in turn, only keeping the desired spectra
    n_fields = 3 * n_zbin
    n_ell = fid_cl.shape[1]
    lmax = n_ell + lmin - 1
    n_spectra = np.sum(keep_spectra)
    covs = np.full((n_ell, n_spectra, n_spectra), np.nan)
    for l in range(lmin, lmax + 1):
        cov_l = calculate_cl_cov_l(fid_cl[:, l - lmin], l, n_fields)[keep_spectra, :][:, keep_spectra]
        covs[l - lmin] = cov_l

    assert np.all(np.isfinite(covs))

    return covs


def get_cl2cf_matrices(theta_bin_edges, lmin, lmax):
    """
    Returns the set of matrices to go from one entire power spectrum to one binned correlation function.

    Args:
        theta_bin_edges (1D numpy array): Angular bin edges in radians.
        lmin (int): Minimum l.
        lmax (int): Maximum l.

    Returns:
        (2D numpy array, \
         2D numpy array, \
         2D numpy array): Tuple of matrices to each go from one entire power spectrum to one binned \
                          correlation function for different spins: (0-0, 2-2, 0-2). The spin-2-2 matrix is only for \
                          xi+, not xi-.
    """

    # Calculate Legendre functions and their derivatives up to lmax
    # pl and dpl indexed as [theta_idx, l]
    cos_thetas = np.cos(theta_bin_edges)
    pl_dpl = np.array([scipy.special.lpn(lmax + 1, cos_theta) for cos_theta in cos_thetas])
    pl = pl_dpl[:, 0, :]
    dpl = pl_dpl[:, 1, :]

    # Calculate various offset combinations of Pl and dPl, and some other useful things
    assert lmin >= 2
    plplus1 = pl[:, (lmin + 1):] # first is l=lmin+1, last is lmax+1
    plminus1 = pl[:, (lmin - 1):lmax] # first is l=lmin-1, last is lmax-1
    xpl = cos_thetas[:, np.newaxis] * pl[:, lmin:(lmax + 1)]
    xdpl = cos_thetas[:, np.newaxis] * dpl[:, lmin:(lmax + 1)]
    dplminus1 = dpl[:, (lmin - 1):lmax]
    xdplminus1 = cos_thetas[:, np.newaxis] * dplminus1
    ell = np.arange(lmin, lmax + 1)
    two_ell_plus1 = 2 * ell + 1
    cos_theta_diff = np.diff(cos_thetas)

    # Calculate bin-averaged Pl, Pl^2 and Gl+/- following Fang et al. eqs 5.6-5.8
    # (Also Friedrich et al. DES Y3 covariance paper, which uses a different sign convention but this cancels out.)
    # All of these vectorised equations have been validated against much slower loop implementations

    # Pl
    pl_bin_top_prediff = plplus1 - plminus1
    pl_bin_top = np.diff(pl_bin_top_prediff, axis=0)
    pl_bin_bottom = np.outer(cos_theta_diff, two_ell_plus1)
    pl_bin = pl_bin_top / pl_bin_bottom

    # Pl^2
    plminus1_coeff = ell + 2 / two_ell_plus1
    plminus1_term = plminus1_coeff[np.newaxis, :] * plminus1
    xpl_coeff = 2 - ell
    xpl_term = xpl_coeff[np.newaxis, :] * xpl
    plplus1_coeff = 2 / two_ell_plus1
    plplus1_term = plplus1_coeff[np.newaxis, :] * plplus1
    pl2_bin_top_prediff = plminus1_term + xpl_term - plplus1_term
    pl2_bin_top = np.diff(pl2_bin_top_prediff, axis=0)
    pl2_bin_bottom = cos_theta_diff[:, np.newaxis]
    pl2_bin = pl2_bin_top / pl2_bin_bottom

    # Gl2+ + Gl2-
    plminus1_coeff = - ell * (ell - 1) / 2 * (ell + 2 / two_ell_plus1) - (ell + 2)
    plminus1_term = plminus1_coeff[np.newaxis, :] * plminus1
    xpl_coeff = - ell * (ell - 1) * (2 - ell) / 2
    xpl_term = xpl_coeff[np.newaxis, :] * xpl
    plplus1_coeff = ell * (ell - 1) / two_ell_plus1
    plplus1_term = plplus1_coeff[np.newaxis, :] * plplus1
    dpl_coeff = 4 - ell
    dpl_term = dpl_coeff * dpl[:, lmin:(lmax + 1)]
    xdplminus1_coeff = ell + 2
    xdplminus1_term = xdplminus1_coeff[np.newaxis, :] * xdplminus1
    xdpl_coeff = 2 * (ell - 1)
    xdpl_term = xdpl_coeff[np.newaxis, :] * xdpl
    pl_coeff = - 2 * (ell - 1)
    pl_term = pl_coeff[np.newaxis, :] * pl[:, lmin:(lmax + 1)]
    dplminus1_coeff = - 2 * (ell + 2)
    dplminus1_term = dplminus1_coeff[np.newaxis, :] * dplminus1
    gplus_bin_top_prediff = (plminus1_term + xpl_term + plplus1_term + dpl_term + xdplminus1_term + xdpl_term + pl_term
                             + dplminus1_term)
    gplus_bin_top = np.diff(gplus_bin_top_prediff, axis=0)
    gplus_bin_bottom = cos_theta_diff[:, np.newaxis]
    gplus_bin = gplus_bin_top / gplus_bin_bottom

    # Apply relevant prefactors to obtain bin-averaged Wigner d symbols
    ell_ellplus1 = (ell * (ell + 1))[np.newaxis, :]
    d00_bin = pl_bin
    d22plus_bin = 2 / ell_ellplus1 ** 2 * gplus_bin
    d02_bin = 1 / ell_ellplus1 * pl2_bin

    # Apply final Wigner prefactor to obtain Cl->CF matrices
    prefac = (two_ell_plus1 / (4 * np.pi))[np.newaxis, :]
    cl2cf_00 = prefac * d00_bin
    cl2cf_22plus = prefac * d22plus_bin
    cl2cf_02 = prefac * d02_bin

    return cl2cf_00, cl2cf_22plus, cl2cf_02


def get_full_cl2cf_matrix(n_zbin, lmax_in, lmin, l_extrap_to, theta_min, theta_max, n_theta_bin):
    """
    Return the primary and secondary transformation matrices to go from a full data vector of power spectra to the
    equivalent data vector of binned correlation functions.

    The primary matrix is to be applied to the observed and model power spectrum.
    The secondary matrix is to be applied to the fiducial power spectrum to obtain a stability vector, which should be
    added to the result of the primary matrix times the observed/model power spectrum to stabilise the sum over l.

    Args:
        n_zbin (int): Number of redshift bins. Three fields per redshift bin are assumed: position, shear E, shear B.
        lmax_in (int): Maximum l in the input to the transformation matrix.
        lmin (int): Minimum l in the input to the transformation matrix.
        l_extrap_to (int): l to extrapolate input power spectra to to obtain stable correlation functions (won't affect
                           likelihood results as long as a consistent value is used).
        theta_min (float): Minimum theta value in radians.
        theta_max (float): Maximum theta value in radians.
        n_theta_bin (int): Number of log-spaced theta bins.

    Returns:
        (2D numpy array, 2D numpy array): Tuple of (primary transformation matrix, secondary transformation matrix), \
                                          each with shape (n_cf * n_theta_bin, n_spectra * n_ell).
    """

    # Generate primary 'extrapolation' matrix for a single power spectrum,
    # which is just the identity matrix padded with zeros
    ident = np.identity(lmax_in - lmin + 1)
    zero = np.zeros((l_extrap_to - lmax_in, lmax_in - lmin + 1))
    pri_extrap_mat = np.block([[ident], [zero]])

    # Generate secondary extrapolation matrix, following equations in weekly notes 21-02-17
    # but with zeros below lmax_in
    zero_top = np.zeros((lmax_in - lmin + 1, lmax_in - lmin + 1))
    zero_bottom = np.zeros((l_extrap_to - lmax_in, lmax_in - lmin + 1 - 2))
    ell_extrap = np.arange(lmax_in + 1, l_extrap_to + 1)
    penul_col = (-ell_extrap + lmax_in) * lmax_in * (lmax_in - 1) / (ell_extrap * (ell_extrap + 1))
    final_col = (ell_extrap - lmax_in + 1) * lmax_in * (lmax_in + 1) / (ell_extrap * (ell_extrap + 1))
    sec_extrap_mat = np.block([[zero_top], [zero_bottom, penul_col[:, np.newaxis], final_col[:, np.newaxis]]])

    # Calculate theta range
    theta_bin_edges = np.logspace(np.log10(theta_min), np.log10(theta_max), n_theta_bin + 1)

    # Calculate bin-averaged individual Cl->CF matrices
    cl2cf_00, cl2cf_22plus, cl2cf_02 = get_cl2cf_matrices(theta_bin_edges, lmin, l_extrap_to)

    # Multiply to obtain combined Cl -> extrapolated Cl -> CF matrices
    pri_cl2cf_00 = cl2cf_00 @ pri_extrap_mat
    sec_cl2cf_00 = cl2cf_00 @ sec_extrap_mat
    pri_cl2cf_22plus = cl2cf_22plus @ pri_extrap_mat
    sec_cl2cf_22plus = cl2cf_22plus @ sec_extrap_mat
    pri_cl2cf_02 = cl2cf_02 @ pri_extrap_mat
    sec_cl2cf_02 = cl2cf_02 @ sec_extrap_mat
    del pri_extrap_mat
    del sec_extrap_mat
    del cl2cf_00
    del cl2cf_22plus
    del cl2cf_02

    # Most blocks in the final combined matrix are just zero
    zero = np.zeros_like(pri_cl2cf_00)

    # Generate list of diagonal-ordered spectra excluding all B-modes except BB
    fields = [f'{f}{z}' for z in range(n_zbin) for f in ('N', 'E', 'B')]
    n_field = len(fields)
    spectra = [fields[row] + fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]
    spectra = [s for s in spectra if 'B' not in s or (s[0] == 'B' and s[2] == 'B')]
    n_spectra = len(spectra)

    # Generate corresponding list of correlation functions, including NN, + and NE
    get_cf = lambda f1, z1, f2, z2: ('++' if f1 + f2 == 'EE' else f1 + f2) + z1 + z2
    cfs = [get_cf(*s) for s in spectra if 'B' not in s]
    n_cf = len(cfs)

    # Preallocate combined transformation matrices
    n_ell_in = lmax_in - lmin + 1
    pri_transmat = np.full((n_cf * n_theta_bin, n_spectra * n_ell_in), np.nan)
    sec_transmat = np.copy(pri_transmat)

    # Populate transformation matrices
    for cf_idx, cf in enumerate(cfs):
        for spec_idx, spec in enumerate(spectra):

            # Extract CF/spectrum types and redshift bins
            cf_type = cf[:2]
            i, j = cf[2:]
            spec_type = spec[::2]
            k, n = spec[1::2]

            # If CF is NN, block is only non-zero if spec is NN and (ij = kn or ij = nk)
            if cf_type == 'NN' and spec_type == 'NN' and (i, j) in [(k, n), (n, k)]:
                pri_block = pri_cl2cf_00
                sec_block = sec_cl2cf_00

            # If CF is ++, block is only non-zero if (spec is EE or spec is BB) and (ij = kn or ij = nk)
            elif cf_type == '++' and spec_type in ['EE', 'BB'] and (i, j) in [(k, n), (n, k)]:
                pri_block = pri_cl2cf_22plus
                sec_block = sec_cl2cf_22plus

            # If CF is NE, block is only non-zero if (spec is NE and ij = kn) or (spec is EN and ij = nk)
            elif cf_type == 'NE' and ((spec_type == 'NE' and (i, j) == (k, n))
                                      or (spec_type == 'EN' and (i, j) == (n, k))):
                pri_block = pri_cl2cf_02
                sec_block = sec_cl2cf_02

            # If CF is EN, block is only non-zero if (spec is EN and ij = kn) or (spec is NE and ij = nk)
            elif cf_type == 'EN' and ((spec_type == 'EN' and (i, j) == (k, n))
                                      or (spec_type == 'NE' and (i, j) == (n, k))):
                pri_block = pri_cl2cf_02
                sec_block = sec_cl2cf_02

            else:
                pri_block = zero
                sec_block = zero

            # Insert blocks into matrices
            start_row = cf_idx * n_theta_bin
            stop_row = start_row + n_theta_bin
            start_col = spec_idx * n_ell_in
            stop_col = start_col + n_ell_in
            pri_transmat[start_row:stop_row, start_col:stop_col] = pri_block
            sec_transmat[start_row:stop_row, start_col:stop_col] = sec_block

    assert np.all(np.isfinite(pri_transmat))
    assert np.all(np.isfinite(sec_transmat))
    return pri_transmat, sec_transmat


def get_cf_noise_variance(n_zbin, theta_min, theta_max, n_theta_bin, survey_area_sqdeg, gals_per_sqarcmin_per_zbin,
                          sigma_e):
    """
    Calculate and return the correlation function noise variance vector, which is the diagonal of the noise contribution
    to the covariance matrix. Off-diagonal elements are zero.

    Args:
        n_zbin (int): Number of redshift bins, assuming 3 fields per bin: position, shear E, shear B.
        theta_min (float): Minimum theta value in radians.
        theta_max (float): Maximum theta value in radians.
        n_theta_bin (int): Number of log-spaced theta bins.
        survey_area_sqdeg (float): Survey area in square degrees.
        gals_per_sqarcmin_per_zbin (float): Number of galaxies per square arcminute per redshift bin (fixed across
                                            all redshift bins).
        sigma_e (float): Intrinsic galaxy ellipticity dispersion per component.

    Returns:
        1D numpy array: Noise variance vector, to be added to the diagonal of the correlation function \
                        covariance matrix.
    """

    # Generate list of diagonal-ordered spectra excluding all B-modes except BB
    fields = [f'{f}{z}' for z in range(n_zbin) for f in ('N', 'E', 'B')]
    n_field = len(fields)
    spectra = [fields[row] + fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]
    spectra = [s for s in spectra if 'B' not in s or (s[0] == 'B' and s[2] == 'B')]

    # Generate corresponding list of correlation functions, including NN, + and NE
    get_cf = lambda f1, z1, f2, z2: ('++' if f1 + f2 == 'EE' else f1 + f2) + z1 + z2
    cfs = [get_cf(*s) for s in spectra if 'B' not in s]

    # Calculate number of galaxy pairs per theta bin
    survey_area_sterad = survey_area_sqdeg * (DEG_TO_RAD ** 2)
    gals_per_sterad = gals_per_sqarcmin_per_zbin * (60 / DEG_TO_RAD) ** 2
    theta_bin_edges = np.logspace(np.log10(theta_min), np.log10(theta_max), n_theta_bin + 1)
    cos_theta = np.cos(theta_bin_edges)
    bin_area_new = 2 * np.pi * -1 * np.diff(cos_theta)
    n_pairs = 0.5 * survey_area_sterad * bin_area_new * (gals_per_sterad ** 2) # Friedrich et al. eq 65
    sigma_4 = sigma_e ** 4

    # Evaluate the noise variance equations
    def noise_variance(cf):
        cf_type = cf[:2]
        z1 = cf[2]
        z2 = cf[3]
        var = 1 / n_pairs
        if cf_type in ('++', 'NE', 'EN'):
            var *= sigma_4
        if cf_type in ('NN', '++') and z1 == z2:
            var *= 2
        return var

    noise_var = np.ravel(np.array(list(map(noise_variance, cfs))))
    return noise_var


def setup(n_zbin, obs_path, fid_pos_pos_dir, fid_she_she_dir, fid_pos_she_dir, lmax, theta_min, theta_max, n_theta_bin,
          survey_area_sqdeg, gals_per_sqarcmin_per_zbin, sigma_e, cov_fsky=1, cl_covs=None, return_cl_covs=False):
    """
    Load and precompute everything that is fixed throughout parameter space. This should be called once per analysis,
    prior to any calls to execute.

    The cl_covs and return_cl_covs options are designed for repeated likelihood runs with the same set of
    (n_zbin, fiducial Cls, lmax), to prevent the unnecessary repeated computation of the Cl covariance.
    On the first run, set (cl_covs=None, return_cl_covs=True), then the function will return (config, cl_covs).
    The cl_covs can then be passed back to setup in future runs.

    Args:
        n_zbin (int): Number of redshift bins. Three fields per redshift bin are assumed: position, shear E and shear B.
        obs_path (str): Path to observed power spectra in numpy .npz file containing two arrays: ell containining the
                        ell range, and obs_cls containing the observed Cls including B-modes, with shape
                        (n_spectra, n_ell), with spectra in diagonal-major order.
        fid_pos_pos_dir (str): Path to the directory containing fiducial position-position power spectra.
        fid_she_she_dir (str): Path to the directory containing fiducial shear-shear power spectra.
        fid_pos_she_dir (str): Path to the directory containing fiducial position-shear power spectra.
        lmax (int): Maximum l value.
        theta_min (float): Minimum theta in radians.
        theta_max (float): Maximum theta in radians.
        n_theta_bin (int): Number of log-spaced theta bins.
        survey_area_sqdeg (float): Survey area in square degrees.
        gals_per_sqarcmin_per_zbin (float): Number of galaxies per square arcminute per redshift bin (fixed across
                                            all redshift bins).
        sigma_e (float): Intrinsic galaxy ellipticity dispersion per component.
        cov_fsky (float, optional): Sky fraction to multiply the entire covariance by 1/fsky (default 1).
        cl_covs (3D numpy array, optional): Power spectrum covariance returned from a previous call to setup with the
                                            same values of n_zbin, lmax, and all fiducial Cls. Calculated if not
                                            supplied.
        return_cl_covs (bool, optional): If True, return the power spectrum covariance so that it can be passed to the
                                         cl_covs option in a future call to setup. (Default False.)

    Returns:
        dict if return_cl_covs is False, \
        else (dict, 3D numpy array): Config dictionary to pass to execute. If return_cl_covs is True, also returns the \
                                     power spectrum covariance which can be passed to the cl_covs option in a future \
                                     call to setup.
    """

    # Generate list of decomposed fields
    print('Generating list of fields')
    field_n = lambda zbin: Field(field_type=FieldType.POSITION, zbin=zbin)
    field_e = lambda zbin: Field(field_type=FieldType.SHEAR_E, zbin=zbin)
    field_b = lambda zbin: Field(field_type=FieldType.SHEAR_B, zbin=zbin)
    decomp_fields = [new_field(zbin) for zbin in range(1, n_zbin + 1) for new_field in (field_n, field_e, field_b)]

    # Generate list of all spectra in the correct (diagonal) order
    print('Generating list of spectra')
    n_decomp_field = len(decomp_fields)
    spectra = [PowerSpectrum(decomp_fields[row], decomp_fields[row + diag])
               for diag in range(n_decomp_field) for row in range(n_decomp_field - diag)]

    # Obtain mask of spectra to keep, which is all except (EB, NB)
    keep_spectra = list(map(keep_spectrum, spectra))
    n_spectra = np.sum(keep_spectra)
    assert n_spectra == (2 * n_zbin) * (2 * n_zbin + 1) // 2 + n_zbin * (n_zbin + 1) // 2

    # Load all obs Cls & ells, and do some consistency checks
    with np.load(obs_path) as data:
        obs_ell = data['ell']
        obs_cl = data['obs_cls']
    n_decomp_fields = 3 * n_zbin
    n_spectra_total = n_decomp_fields * (n_decomp_fields + 1) // 2
    if obs_ell[0] > 0:
        obs_cl = np.concatenate((np.zeros((n_spectra_total, obs_ell[0])), obs_cl), axis=1)
        obs_ell = np.concatenate((np.arange(obs_ell[0]), obs_ell))
    assert obs_cl.shape[0] == n_spectra_total
    assert obs_ell[0] == 0
    obs_lmax = np.amax(obs_ell)
    assert np.all(np.arange(obs_lmax + 1) == obs_ell)
    if lmax <= obs_lmax:
        obs_cl = obs_cl[:, :(lmax + 1)]
    else:
        raise ValueError(f'obs lmax ({obs_lmax}) is less than likelihood lmax ({lmax})')
    lmin = 2
    obs_cl = obs_cl[:, lmin:]

    # Throw away all spectra including B modes except BB and reshape into a vector grouped by spectrum
    obs_cl = obs_cl[keep_spectra, :].flatten()
    n_ell = lmax - lmin + 1
    assert obs_cl.shape == (n_spectra * n_ell,)

    # Load fiducial Cls and ells with zero B-modes
    fid_ell_pos = np.loadtxt(os.path.join(fid_pos_pos_dir, 'ell.txt'))
    fid_ell_she = np.loadtxt(os.path.join(fid_she_she_dir, 'ell.txt'))
    fid_ell_shp = np.loadtxt(os.path.join(fid_pos_she_dir, 'ell.txt'))
    assert np.allclose(fid_ell_pos, fid_ell_she)
    assert np.allclose(fid_ell_pos, fid_ell_shp)
    fid_lmin = int(np.amin(fid_ell_pos))
    fid_lmax = int(np.amax(fid_ell_pos))
    assert np.allclose(fid_ell_pos, np.arange(fid_lmin, fid_lmax + 1))
    fid_cl, _ = load_cls_zerob(n_zbin, fid_pos_pos_dir, fid_she_she_dir, fid_pos_she_dir, lmax, fid_lmin)
    fid_cl = fid_cl[:, lmin:]

    # Calculate primary and secondary transformation matrices which take you from Cls to binned CFs
    l_extrap_to = 60000
    transmat, sec_transmat = get_full_cl2cf_matrix(n_zbin, lmax, lmin, l_extrap_to, theta_min, theta_max, n_theta_bin)
    assert np.all(np.isfinite(transmat))
    assert np.all(np.isfinite(sec_transmat))

    # Apply secondary transformation matrix to fiducial Cl vector to obtain stabilisation vector
    fid_cl_vec = np.ravel(fid_cl[keep_spectra, :])
    stabl_vec = sec_transmat @ fid_cl_vec
    assert np.all(np.isfinite(stabl_vec))

    # Calculate per-l Cl covariance matrices for fiducial Cls
    if cl_covs is None:
        cl_covs = calculate_cl_covs(fid_cl, n_zbin, keep_spectra, lmin)
    assert np.all(np.isfinite(cl_covs))

    # Calculate CF cov with einsum
    # This requires that transmat is first reshaped so it can be indexed as [cf_idx, spec_idx, ell_idx]
    n_cf, _ = transmat.shape
    n_ell, n_spectra, _ = cl_covs.shape
    transmat_3d = np.reshape(transmat, (n_cf, n_spectra, n_ell))
    cf_cov = np.einsum('ikl,jnl,lkn->ij', transmat_3d, transmat_3d, cl_covs, optimize='greedy')

    # Add noise variance
    cf_noise_var = get_cf_noise_variance(n_zbin, theta_min, theta_max, n_theta_bin, survey_area_sqdeg,
                                         gals_per_sqarcmin_per_zbin, sigma_e)
    cf_noise_cov = np.diag(cf_noise_var)
    cf_cov = cf_cov + cf_noise_cov
    assert np.all(np.linalg.eigvals(cf_cov) > 0)

    # Apply 1/fsky factor
    cf_cov *= 1. / cov_fsky

    # Do some checks
    assert np.all(np.isfinite(cf_cov))
    assert np.allclose(cf_cov, cf_cov.T)
    assert np.all(np.linalg.eigvals(cf_cov) > 0)
    assert np.all(np.diag(cf_cov) > 0)

    # Extract variance and correlation
    cf_std = np.sqrt(np.diag(cf_cov))
    cf_corr = cf_cov / np.outer(cf_std, cf_std)
    assert np.all(np.isreal(cf_corr))
    assert np.all(np.abs(cf_corr) < 1 + 1e-6)
    assert np.all(np.linalg.eigvals(cf_corr) > 0)

    # Elementwise multiply transformation matrix and stabilisation vector by std
    transmat /= cf_std[:, np.newaxis]
    assert np.all(np.isfinite(transmat))
    stabl_vec /= cf_std
    assert np.all(np.isfinite(stabl_vec))

    # Transform observation
    obs_cf = transmat @ obs_cl + stabl_vec
    assert np.all(np.isfinite(obs_cf))

    # Now just use the correlation matrix in place of the covariance matrix
    cf_cov = cf_corr

    # Form config dictionary
    config = {
        'obs_cf': obs_cf,
        'keep_spectra': keep_spectra,
        'transmat': transmat,
        'cf_cov': cf_cov,
        'stabl_vec': stabl_vec,
        'lmin': lmin
    }

    if return_cl_covs:
        return config, cl_covs

    return config


def execute(theory_cl, config):
    """
    Evaluate log-likelihood at theory_cl.

    Args:
        theory_cl (2D numpy array): Theory power spectra, as loaded by load_cls_zerob with shape (n_spectra, lmax + 1).
        config (dict): Config dictionary returned by setup.

    Returns:
        float: Log-likelihood value.
    """

    # Pull fixed (model Cl-independent) parameters from config
    obs_cf = config['obs_cf']
    keep_spectra = config['keep_spectra']
    transmat = config['transmat']
    cf_cov = config['cf_cov']
    stabl_vec = config['stabl_vec']
    lmin = config['lmin']

    # Throw away all spectra including B modes except BB and reshape into a vector grouped by spectrum
    theory_cl = np.ravel(theory_cl[keep_spectra, lmin:])

    # Transform to vector of correlation functions
    theory_cf = transmat @ theory_cl + stabl_vec
    assert theory_cf.shape == obs_cf.shape

    # Evaluate the likelihood
    log_like = scipy.stats.multivariate_normal.logpdf(obs_cf, theory_cf, cf_cov)
    return log_like
