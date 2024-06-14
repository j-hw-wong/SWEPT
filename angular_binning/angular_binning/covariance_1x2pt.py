"""
Functions to calculate 3x2pt Gaussian covariance.
"""

import enum
import time
import warnings
from collections import namedtuple

import gaussian_cl_likelihood.python.simulation # https://github.com/robinupham/gaussian_cl_likelihood
import healpy as hp
import numpy as np
import pymaster as nmt

warnings.filterwarnings('error')  # terminate on warning


# Simple classes to help keep this understandable
Field = namedtuple('Field', ['field_type', 'zbin'])
PowerSpectrum = namedtuple('PowerSpectrum', ['field_1', 'field_2'])


class FieldType(enum.Enum):
    """
    The three types of fields: position, shear E and shear B.
    """
    POSITION = enum.auto()
    #JW - ?? Throughout the code only two types are considered - POSITION and SHEAR, i.e. no
    # distinction between E- or B-mode.. How to handle this?
    SHEAR = enum.auto()
    SHEAR_E = enum.auto()
    SHEAR_B = enum.auto()


def spin_from_field(field):
    """
    Returns the spin of a field.

    Args:
        field (Field): Field to get the spin of.

    Returns:
        int: The spin of the field.
    """

    if field.field_type == FieldType.POSITION:
        return 0
    elif field.field_type == FieldType.SHEAR:
        return 2
    else:
        raise ValueError(f'Field type {field.field_type} is not POSITION or SHEAR')


def spins_from_spectrum(spec):
    """
    Returns the two spins corresponding to the two fields of the supplied power spectrum.

    Args:
        spec (PowerSpectrum): Power spectrum to get the spins of.

    Returns:
        (int, int): The spins of the two fields of the power spectrum.
    """

    fields = (spec.field_1, spec.field_2)
    return map(spin_from_field, fields)


def spectrum_from_fields(field_a, field_b):
    """
    Returns a PowerSpectrum object corresponding to the supplied fields, in a consistent order, which is
    lower redshift bin first, and if the bins are equal then position first.

    (Note that this is not the same convention as used when loading from CosmoSIS output.)

    Args:
        field_a (Field): One field in the spectrum.
        field_b (Field): The other field in the spectrum.

    Returns:
        PowerSpectrum: Power spectrum corresponding to the two fields.
    """

    if field_a.zbin < field_b.zbin or (field_a.zbin == field_b.zbin and field_a.field_type == FieldType.POSITION):
        return PowerSpectrum(field_1=field_a, field_2=field_b)
    return PowerSpectrum(field_1=field_b, field_2=field_a)


def workspace_from_spins(spin_a, spin_b, workspace_spin00, workspace_spin02, workspace_spin22):
    """
    Returns the appropriate NmtWorkspace object for the two supplied spins.

    Args:
        spin_a (int): Spin of one field.
        spin_2 (int): Spin of the other field.
        workspace_spin00 (NmtWorkspace): Workspace for two spin-0 fields.
        workspace_spin02 (NmtWorkspace): Workspace for one spin-0 and one spin-2 field (in either order).
        workspace_spin22 (NmtWorkspace): Workspace for two spin-2 fields.

    Returns:
        NmtWorkspace: The workspace object corresponding to the two supplied spins.
    """

    spins = (spin_a, spin_b)
    if spins == (0, 0):
        return workspace_spin00
    if spins in [(0, 2), (2, 0)]:
        return workspace_spin02
    if spins == (2, 2):
        return workspace_spin22
    raise ValueError(f'Unexpected combination of spins {spins}')


def load_cls(signal_paths, noise_paths, lmax_in, lmax_out, signal_lmin, noise_lmin):
    """
    Load a list of Cls with appropriate noise, given lists of path to signal and noise Cls. If any entry in either list
    is None, then zeros are used. Signal and noise are padded with zeros below ``signal_lmin`` and ``noise_lmin`` and
    above ``lmax_in`` (if less than ``lmax_out``).

    Args:
        signal_paths (list): List of paths to signal Cls. If any entry is None, it is taken to be zero.
        noise_paths (list): List of paths to noise Cls. If any entry is None, it is taken to be zero.
        lmax_in (int): Maximum l to load. If this is less than ``lmax_out`` then padded with zeros above this l.
        lmax_out (int): Maximum l to return.
        signal_lmin (int): First l of signal power spectra. Signal Cls will be padded with zeros below this.
        noise_lmin (int): First l of noise power spectra. Noise Cls will be padded with zeros below this. If None,
        assumes no noise spectra to be combined with signal (e.g. if calculating signal and noise covariance separately)
        ^NOTE: Might need to write a guard for this...

    Returns:
        list: List of numpy arrays, each of which is a signal + noise power spectrum in the supplied order.
    """

    if noise_lmin is None:
        assert all(n is None for n in noise_paths)

    # If a signal or noise path is None then just use zeros
    zero_cl = np.zeros(lmax_out + 1)
    zero_pad = np.zeros(lmax_out - lmax_in) if lmax_out > lmax_in else []

    if lmax_in > lmax_out:
        lmax_in = lmax_out

    # Load Cls with appropriate padding and add signal and noise
    combined_cls = []
    for signal_path, noise_path in zip(signal_paths, noise_paths):

        signal_cl = (np.concatenate((np.zeros(signal_lmin),
                                     np.loadtxt(signal_path, max_rows=(lmax_in - signal_lmin + 1)), zero_pad))
                     if signal_path else zero_cl)
        noise_cl = (np.concatenate((np.zeros(noise_lmin),
                                    np.loadtxt(noise_path, max_rows=(lmax_in - noise_lmin + 1)), zero_pad))
                    if noise_path else zero_cl)
        combined_cls.append(signal_cl + noise_cl)

    return combined_cls


def get_1x2pt_cov_signal_pcl(n_zbin, pos_pos_filemask, pos_she_filemask, she_she_filemask, field, lmax_in, lmin_in,
                             lmax_out, lmin_out, mask_path, nside, save_filemask):
    """
    Calculate 3x2pt Gaussian covariance using NaMaster, saving each block separately to disk. This function computes
    the covariance associated with a Pseudo-Cl in the absence of noise - i.e. we apply the coupling coefficients of
    a mask to a theoretical (full-sky) Cl (using the coupled=True argument in NaMaster gaussian_covariance). The
    complete covariance for a noisy Pseudo-Cl observation can be calculated by combining this function with
    get_3x2pt_cov_noise - which calculates the Gaussian covariance of a predicted noise spectrum. NOTE - this
    theoretical noise spectrum must predict the noise associated with the region of sky observed (unlike the full-sky
    Cl) since it is NOT combined with the mask coupling coefficients.

    Args:
        n_zbin (int): Number of redshift bins, assuming 1 position field and 1 shear field per redshift bin.
        pos_pos_filemask (str): Path to text file containing a position-position power spectrum with ``{hi_zbin}`` and
                                ``{lo_zbin}`` placeholders.
        pos_she_filemask (str): Path to text file containing a position-shear power spectrum with ``{pos_zbin}`` and
                                ``{she_zbin}`` placeholders.
        she_she_filemask (str): Path to text file containing a shear-shear power spectrum with ``{hi_zbin}`` and
                                ``{lo_zbin}`` placeholders.
        lmax_in (int): Maximum l to including in mixing.
        lmin_in (int): Minimum l in input power spectra.
        lmax_out (int): Maximum l to include in covariance.
        lmin_out (int): Minimum l to include in covariance.
        noise_lmin (int): Minimum l in noise power spectra.
        mask_path (str): Path to mask FITS file. If None, full sky is assumed, in which case the covariance will be
                         diagonal.
        nside (int): HEALPix resolution nside parameter.
        save_filemask (str): Path to save each covariance block to disk, with ``{spec1_idx}`` and ``{spec2_idx}``
                             placeholders.
    """

    # Load and rescale mask, and calculate fsky
    if mask_path is not None:
        print('Loading and rescaling mask')
        mask = hp.pixelfunc.ud_grade(hp.read_map(mask_path, dtype=float), nside)
        assert np.amin(mask) == 0
        assert np.amax(mask) == 1
    else:
        print('Full sky')
        mask = np.ones(hp.pixelfunc.nside2npix(nside))
    assert np.all(np.isfinite(mask))
    fsky = np.mean(mask)
    print(f'fsky = {fsky:.3f}')

    # Create NaMaster binning scheme as individual Cls
    print('Creating binning scheme')
    bins = nmt.NmtBin.from_lmax_linear(lmax_in, 1)

    # Calculate mixing matrices for spin 0-0, 0-2 (equivalent to 2-0), and 2-2
    field_spin0 = nmt.NmtField(mask, None, spin=0, lite=True, lmax_sht=lmax_in)
    field_spin2 = nmt.NmtField(mask, None, spin=2, lite=True, lmax_sht=lmax_in)
    # field_spin0 = nmt.NmtField(mask, None, spin=0, lite=True)
    # field_spin2 = nmt.NmtField(mask, None, spin=2, lite=True)
    workspace_spin00 = nmt.NmtWorkspace()
    print(f'Calculating mixing matrix 1 / 3 at {time.strftime("%c")}')
    workspace_spin00.compute_coupling_matrix(field_spin0, field_spin0, bins)
    workspace_spin02 = nmt.NmtWorkspace()
    print(f'Calculating mixing matrix 2 / 3 at {time.strftime("%c")}')
    workspace_spin02.compute_coupling_matrix(field_spin0, field_spin2, bins)
    workspace_spin22 = nmt.NmtWorkspace()
    print(f'Calculating mixing matrix 3 / 3 at {time.strftime("%c")}')
    workspace_spin22.compute_coupling_matrix(field_spin2, field_spin2, bins)
    assert np.all(np.isfinite(workspace_spin00.get_coupling_matrix()))
    assert np.all(np.isfinite(workspace_spin02.get_coupling_matrix()))
    assert np.all(np.isfinite(workspace_spin22.get_coupling_matrix()))

    # Generate list of fields
    print('Generating list of fields')
    new_pos_field = lambda zbin: Field(field_type=FieldType.POSITION, zbin=zbin)
    new_she_field = lambda zbin: Field(field_type=FieldType.SHEAR, zbin=zbin)

    if field == 'E':
        fields = [new_she_field(zbin) for zbin in range(1, n_zbin + 1)]
        print(fields)

    else:
        assert field == 'N'
        fields = [new_pos_field(zbin) for zbin in range(1, n_zbin + 1)]
        print(fields)

    # Generate list of target spectra (the ones we want the covariance for) in the correct (diagonal) order
    print('Generating list of spectra')
    n_field = len(fields)
    spectra = [PowerSpectrum(fields[row], fields[row + diag])
               for diag in range(n_field) for row in range(n_field - diag)]

    # Generate list of sets of mode-coupled theory Cls corresponding to the target spectra
    coupled_theory_cls = []
    for spec_idx, spec in enumerate(spectra):
        print(f'Loading and coupling input Cls {spec_idx + 1} / {len(spectra)}')

        field_types = (spec.field_1.field_type, spec.field_2.field_type)
        print(field_types)
        zbins = (spec.field_1.zbin, spec.field_2.zbin)

        # Get paths of signal and noise spectra to load
        if field_types == (FieldType.POSITION, FieldType.POSITION):
            # NN only
            signal_paths = [pos_pos_filemask.format(hi_zbin=max(zbins), lo_zbin=min(zbins))]
            noise_paths = [None]
        elif field_types == (FieldType.POSITION, FieldType.SHEAR):
            # NE, NB
            signal_paths = [pos_she_filemask.format(pos_zbin=zbins[0], she_zbin=zbins[1]), None]
            noise_paths = [None, None]
        elif field_types == (FieldType.SHEAR, FieldType.POSITION):
            # EN, BN
            signal_paths = [pos_she_filemask.format(pos_zbin=zbins[1], she_zbin=zbins[0]), None]
            noise_paths = [None, None]
        elif field_types == (FieldType.SHEAR, FieldType.SHEAR):
            # EE, EB, BE, BB
            signal_paths = [she_she_filemask.format(hi_zbin=max(zbins), lo_zbin=min(zbins)), None, None, None]
            noise_paths = [None, None, None, None]
        else:
            raise ValueError(f'Unexpected field type combination: {field_types}')
        # Load in the signal + noise Cls
        print(f'Loading and coupling input Cls {spec_idx + 1} / {len(spectra)}: Loading...')
        uncoupled_theory_cls = load_cls(signal_paths, noise_paths, lmax_in, lmax_in, lmin_in, noise_lmin=None)

        # Apply the "improved NKA" method: couple the theory Cls,
        # then divide by fsky to avoid double-counting the reduction in power
        print(f'Loading and coupling input Cls {spec_idx + 1} / {len(spectra)}: Coupling...')
        spins = tuple(spins_from_spectrum(spec))

        if spins == (0, 0):
            assert len(uncoupled_theory_cls) == 1
            workspace = workspace_spin00
        elif spins in ((0, 2), (2, 0)):
            assert len(uncoupled_theory_cls) == 2
            workspace = workspace_spin02
        elif spins == (2, 2):
            assert len(uncoupled_theory_cls) == 4
            workspace = workspace_spin22
        else:
            raise ValueError(f'Unexpected spins: {spins}')
        coupled_cls = workspace.couple_cell(uncoupled_theory_cls)
        assert np.all(np.isfinite(coupled_cls))
        assert len(coupled_cls) == len(uncoupled_theory_cls)
        coupled_theory_cls.append(np.divide(coupled_cls, fsky))

    # Calculate additional covariance coupling coefficients (independent of spin)
    print(f'Computing covariance coupling coefficients at {time.strftime("%c")}')
    cov_workspace = nmt.NmtCovarianceWorkspace()
    cov_workspace.compute_coupling_coefficients(field_spin2, field_spin2, lmax=lmax_in)

    # Iterate over unique pairs of spectra
    for spec_a_idx, spec_a in enumerate(spectra):

        # Obtain the spins and workspace for the first spectrum
        spin_a1, spin_a2 = spins_from_spectrum(spec_a)
        workspace_a = workspace_from_spins(spin_a1, spin_a2, workspace_spin00, workspace_spin02, workspace_spin22)

        for spec_b_idx, spec_b in enumerate(spectra[:(spec_a_idx + 1)]):
            print(f'Calculating covariance block row {spec_a_idx} column {spec_b_idx} at {time.strftime("%c")}')

            # Obtain the spins and workspace for the second spectrum
            spin_b1, spin_b2 = spins_from_spectrum(spec_b)
            workspace_b = workspace_from_spins(spin_b1, spin_b2, workspace_spin00, workspace_spin02, workspace_spin22)

            # Identify the four power spectra we need to calculate this covariance
            a1b1 = spectrum_from_fields(spec_a.field_1, spec_b.field_1)
            a1b2 = spectrum_from_fields(spec_a.field_1, spec_b.field_2)
            a2b1 = spectrum_from_fields(spec_a.field_2, spec_b.field_1)
            a2b2 = spectrum_from_fields(spec_a.field_2, spec_b.field_2)

            # Obtain the corresponding theory Cls
            cl_a1b1 = coupled_theory_cls[spectra.index(a1b1)]
            cl_a1b2 = coupled_theory_cls[spectra.index(a1b2)]
            cl_a2b1 = coupled_theory_cls[spectra.index(a2b1)]
            cl_a2b2 = coupled_theory_cls[spectra.index(a2b2)]
            assert np.all(np.isfinite(cl_a1b1))
            assert np.all(np.isfinite(cl_a1b2))
            assert np.all(np.isfinite(cl_a2b1))
            assert np.all(np.isfinite(cl_a2b2))

            # Evaluate the covariance
            cl_cov = nmt.gaussian_covariance(cov_workspace, spin_a1, spin_a2, spin_b1, spin_b2, cl_a1b1, cl_a1b2,
                                             cl_a2b1, cl_a2b2, workspace_a, workspace_b, coupled=True)

            # Extract the part of the covariance we want, which is conveniently always the [..., 0, ..., 0] block,
            # since all other blocks relate to B-modes
            cl_cov = cl_cov.reshape((lmax_in + 1, len(coupled_theory_cls[spec_a_idx]),
                                     lmax_in + 1, len(coupled_theory_cls[spec_b_idx])))
            cl_cov = cl_cov[:, 0, :, 0]
            cl_cov = cl_cov[lmin_out:(lmax_out + 1), lmin_out:(lmax_out + 1)]

            # Do some checks and save to disk
            assert np.all(np.isfinite(cl_cov))
            n_ell_out = lmax_out - lmin_out + 1
            assert cl_cov.shape == (n_ell_out, n_ell_out)
            if spec_a_idx == spec_b_idx:
                assert np.allclose(cl_cov, cl_cov.T)
            save_path = save_filemask.format(spec1_idx=spec_a_idx, spec2_idx=spec_b_idx)
            header = (f'Output from {__file__}.get_3x2pt_cov for spectra ({spec_a}, {spec_b}), with parameters '
                      f'n_zbin = {n_zbin}, pos_pos_filemask {pos_pos_filemask}, pos_she_filemask {pos_she_filemask}, '
                      f'she_she_filemask {she_she_filemask}, lmax_in = {lmax_in}, lmin_in = {lmin_in}, '
                      f'lmax_out = {lmax_out}, lmin_out = {lmin_out}, '
                      f'mask_path = {mask_path}, '
                      f'nside = {nside}; time {time.strftime("%c")}')
            print(f'Saving block at {time.strftime("%c")}')
            np.savez_compressed(save_path, cov_block=cl_cov, spec1_idx=spec_a_idx, spec2_idx=spec_b_idx, header=header)
            print(f'Saving {save_path} at {time.strftime("%c")}')

    print(f'Done at {time.strftime("%c")}')


def get_1x2pt_cov_noise(n_zbin, lmax_in, lmin_in, lmax_out, lmin_out, pos_nl_path, she_nl_path, pos_she_nl_path, field,
                        noise_lmin, nside, save_filemask, mask_path=None):
    """
    Calculate 3x2pt Gaussian covariance using NaMaster, saving each block separately to disk.

    Args:
        n_zbin (int): Number of redshift bins, assuming 1 position field and 1 shear field per redshift bin.
        #pos_pos_filemask (str): Path to text file containing a position-position power spectrum with ``{hi_zbin}`` and
        #                        ``{lo_zbin}`` placeholders.
        #pos_she_filemask (str): Path to text file containing a position-shear power spectrum with ``{pos_zbin}`` and
        #                        ``{she_zbin}`` placeholders.
        #she_she_filemask (str): Path to text file containing a shear-shear power spectrum with ``{hi_zbin}`` and
        #                        ``{lo_zbin}`` placeholders.
        lmax_in (int): Maximum l to including in mixing.
        lmin_in (int): Minimum l in input power spectra.
        lmax_out (int): Maximum l to include in covariance.
        lmin_out (int): Minimum l to include in covariance.
        pos_nl_path (str): Path to position noise power spectrum.
        she_nl_path (str): Path to shear noise power spectrum.
        noise_lmin (int): Minimum l in noise power spectra.
        mask_path (str): Path to mask FITS file. If None, full sky is assumed, in which case the covariance will be
                         diagonal.
        nside (int): HEALPix resolution nside parameter.
        save_filemask (str): Path to save each covariance block to disk, with ``{spec1_idx}`` and ``{spec2_idx}``
                             placeholders.
    """

    # Load and rescale mask, and calculate fsky
    assert mask_path is None
    #if mask_path is not None:
    #    print('Loading and rescaling mask')
    #    mask = hp.pixelfunc.ud_grade(hp.read_map(mask_path, dtype=float), nside)
    #    assert np.amin(mask) == 0
    #    assert np.amax(mask) == 1
    #else:
    #print('Full sky')
    mask = np.ones(hp.pixelfunc.nside2npix(nside))
    assert np.all(np.isfinite(mask))
    fsky = np.mean(mask)
    #fsky=0.001
    print(f'fsky = {fsky:.3f}')

    # Create NaMaster binning scheme as individual Cls
    print('Creating binning scheme')
    bins = nmt.NmtBin.from_lmax_linear(lmax_out, 1)

    # Calculate mixing matrices for spin 0-0, 0-2 (equivalent to 2-0), and 2-2
    field_spin0 = nmt.NmtField(mask, None, spin=0, lite=True, lmax_sht=lmax_in)
    field_spin2 = nmt.NmtField(mask, None, spin=2, lite=True, lmax_sht=lmax_in)
    # field_spin0 = nmt.NmtField(mask, None, spin=0, lite=True)
    # field_spin2 = nmt.NmtField(mask, None, spin=2, lite=True)
    workspace_spin00 = nmt.NmtWorkspace()
    print(f'Calculating mixing matrix 1 / 3 at {time.strftime("%c")}')
    workspace_spin00.compute_coupling_matrix(field_spin0, field_spin0, bins)
    workspace_spin02 = nmt.NmtWorkspace()
    print(f'Calculating mixing matrix 2 / 3 at {time.strftime("%c")}')
    workspace_spin02.compute_coupling_matrix(field_spin0, field_spin2, bins)
    workspace_spin22 = nmt.NmtWorkspace()
    print(f'Calculating mixing matrix 3 / 3 at {time.strftime("%c")}')
    workspace_spin22.compute_coupling_matrix(field_spin2, field_spin2, bins)
    assert np.all(np.isfinite(workspace_spin00.get_coupling_matrix()))
    assert np.all(np.isfinite(workspace_spin02.get_coupling_matrix()))
    assert np.all(np.isfinite(workspace_spin22.get_coupling_matrix()))

    # Generate list of fields
    print('Generating list of fields')
    new_pos_field = lambda zbin: Field(field_type=FieldType.POSITION, zbin=zbin)
    new_she_field = lambda zbin: Field(field_type=FieldType.SHEAR, zbin=zbin)
    #fields = [new_field(zbin) for zbin in range(1, n_zbin + 1) for new_field in (new_pos_field, new_she_field)]

    if field == 'E':
        fields = [new_she_field(zbin) for zbin in range(1, n_zbin + 1)]
        print(fields)

    else:
        assert field == 'N'
        fields = [new_pos_field(zbin) for zbin in range(1, n_zbin + 1)]
        print(fields)

    # Generate list of target spectra (the ones we want the covariance for) in the correct (diagonal) order
    print('Generating list of spectra')
    n_field = len(fields)
    spectra = [PowerSpectrum(fields[row], fields[row + diag])
               for diag in range(n_field) for row in range(n_field - diag)]

    # Generate list of sets of mode-coupled theory Cls corresponding to the target spectra
    coupled_theory_cls = []
    for spec_idx, spec in enumerate(spectra):
        print(f'Loading and coupling input Cls {spec_idx + 1} / {len(spectra)}')

        field_types = (spec.field_1.field_type, spec.field_2.field_type)
        zbins = (spec.field_1.zbin, spec.field_2.zbin)

        # Get paths of signal and noise spectra to load
        if field_types == (FieldType.POSITION, FieldType.POSITION):
            # NN only
            signal_paths = [None]
            #print(pos_nl_path)
            noise_paths = [pos_nl_path.format(hi_zbin=max(zbins), lo_zbin=min(zbins))]
        elif field_types == (FieldType.POSITION, FieldType.SHEAR):
            # NE, NB
            signal_paths = [None, None]
            noise_paths = [pos_she_nl_path.format(pos_zbin=zbins[0], she_zbin=zbins[1]), None]
        elif field_types == (FieldType.SHEAR, FieldType.POSITION):
            # EN, BN
            signal_paths = [None, None]
            noise_paths = [pos_she_nl_path.format(pos_zbin=zbins[1], she_zbin=zbins[0]), None]
        elif field_types == (FieldType.SHEAR, FieldType.SHEAR):
            # EE, EB, BE, BB
            signal_paths = [None, None, None, None]
            noise_paths = [she_nl_path.format(hi_zbin=max(zbins), lo_zbin=min(zbins)),
                           None,
                           None,
                           she_nl_path.format(hi_zbin=max(zbins), lo_zbin=min(zbins))]
        else:
            raise ValueError(f'Unexpected field type combination: {field_types}')
        #(signal_paths,noise_paths)
        # Load in the signal + noise Cls
        print(f'Loading and coupling input Cls {spec_idx + 1} / {len(spectra)}: Loading...')
        uncoupled_theory_cls = load_cls(signal_paths, noise_paths, lmax_in, lmax_out, lmin_in, noise_lmin)

        # Apply the "improved NKA" method: couple the theory Cls,
        # then divide by fsky to avoid double-counting the reduction in power
        print(f'Loading and coupling input Cls {spec_idx + 1} / {len(spectra)}: Coupling...')
        spins = tuple(spins_from_spectrum(spec))
        if spins == (0, 0):
            assert len(uncoupled_theory_cls) == 1
            workspace = workspace_spin00
        elif spins in ((0, 2), (2, 0)):
            assert len(uncoupled_theory_cls) == 2
            workspace = workspace_spin02
        elif spins == (2, 2):
            assert len(uncoupled_theory_cls) == 4
            workspace = workspace_spin22
        else:
            raise ValueError(f'Unexpected spins: {spins}')
        coupled_cls = workspace.couple_cell(uncoupled_theory_cls)
        assert np.all(np.isfinite(coupled_cls))
        assert len(coupled_cls) == len(uncoupled_theory_cls)
        coupled_theory_cls.append(np.divide(coupled_cls, fsky))

    # Calculate additional covariance coupling coefficients (independent of spin)
    print(f'Computing covariance coupling coefficients at {time.strftime("%c")}')
    cov_workspace = nmt.NmtCovarianceWorkspace()
    cov_workspace.compute_coupling_coefficients(field_spin2, field_spin2, lmax=lmax_out)

    # Iterate over unique pairs of spectra
    for spec_a_idx, spec_a in enumerate(spectra):

        # Obtain the spins and workspace for the first spectrum
        spin_a1, spin_a2 = spins_from_spectrum(spec_a)
        workspace_a = workspace_from_spins(spin_a1, spin_a2, workspace_spin00, workspace_spin02, workspace_spin22)

        for spec_b_idx, spec_b in enumerate(spectra[:(spec_a_idx + 1)]):
            print(f'Calculating covariance block row {spec_a_idx} column {spec_b_idx} at {time.strftime("%c")}')

            # Obtain the spins and workspace for the second spectrum
            spin_b1, spin_b2 = spins_from_spectrum(spec_b)
            workspace_b = workspace_from_spins(spin_b1, spin_b2, workspace_spin00, workspace_spin02, workspace_spin22)

            # Identify the four power spectra we need to calculate this covariance
            a1b1 = spectrum_from_fields(spec_a.field_1, spec_b.field_1)
            a1b2 = spectrum_from_fields(spec_a.field_1, spec_b.field_2)
            a2b1 = spectrum_from_fields(spec_a.field_2, spec_b.field_1)
            a2b2 = spectrum_from_fields(spec_a.field_2, spec_b.field_2)

            # Obtain the corresponding theory Cls
            cl_a1b1 = coupled_theory_cls[spectra.index(a1b1)]
            cl_a1b2 = coupled_theory_cls[spectra.index(a1b2)]
            cl_a2b1 = coupled_theory_cls[spectra.index(a2b1)]
            cl_a2b2 = coupled_theory_cls[spectra.index(a2b2)]
            assert np.all(np.isfinite(cl_a1b1))
            assert np.all(np.isfinite(cl_a1b2))
            assert np.all(np.isfinite(cl_a2b1))
            assert np.all(np.isfinite(cl_a2b2))

            # Evaluate the covariance
            cl_cov = nmt.gaussian_covariance(cov_workspace, spin_a1, spin_a2, spin_b1, spin_b2, cl_a1b1, cl_a1b2,
                                             cl_a2b1, cl_a2b2, workspace_a, workspace_b, coupled=True)

            ############################################################################################################
            # SHOULD CHECK THAT THE COUPLED=TRUE HAS NO EFFECT - ASSUMING A FULL-SKY I.E. NO CONSIDERATION OF
            # MODE-COUPLING IS THE SAME AS JUST COUPLING OVER A MASK OF 1S - SEE IF BELOW IS THE SAME OUTPUT
            ############################################################################################################

            #cl_cov = nmt.gaussian_covariance(cov_workspace, spin_a1, spin_a2, spin_b1, spin_b2, cl_a1b1, cl_a1b2,
            #                                 cl_a2b1, cl_a2b2, workspace_a, workspace_b, coupled=False)

            # Extract the part of the covariance we want, which is conveniently always the [..., 0, ..., 0] block,
            # since all other blocks relate to B-modes
            cl_cov = cl_cov.reshape((lmax_out + 1, len(coupled_theory_cls[spec_a_idx]),
                                     lmax_out + 1, len(coupled_theory_cls[spec_b_idx])))
            cl_cov = cl_cov[:, 0, :, 0]
            cl_cov = cl_cov[lmin_out:(lmax_out + 1), lmin_out:(lmax_out + 1)]

            # Do some checks and save to disk
            assert np.all(np.isfinite(cl_cov))
            n_ell_out = lmax_out - lmin_out + 1
            assert cl_cov.shape == (n_ell_out, n_ell_out)
            if spec_a_idx == spec_b_idx:
                assert np.allclose(cl_cov, cl_cov.T)
            save_path = save_filemask.format(spec1_idx=spec_a_idx, spec2_idx=spec_b_idx)
            header = (f'Output from {__file__}.get_3x2pt_cov for spectra ({spec_a}, {spec_b}), with parameters '
                      f'n_zbin = {n_zbin}, '
                      f'lmax_in = {lmax_in}, lmin_in = {lmin_in}, '
                      f'lmax_out = {lmax_out}, lmin_out = {lmin_out}, pos_nl_path = {pos_nl_path},  pos_she_nl_path = {pos_she_nl_path},'
                      f'she_nl_path = {she_nl_path}, noise_lmin = {noise_lmin}, mask_path = {mask_path}, '
                      f'nside = {nside}; time {time.strftime("%c")}')
            print(f'Saving block at {time.strftime("%c")}')
            np.savez_compressed(save_path, cov_block=cl_cov, spec1_idx=spec_a_idx, spec2_idx=spec_b_idx, header=header)
            print(f'Saving {save_path} at {time.strftime("%c")}')

    print(f'Done at {time.strftime("%c")}')


def bin_combine_cov(n_zbin, lmin_in, lmin_out, lmax, n_bp_min, n_bp_max, input_filemask, save_filemask,
                    bandpower_spacing='log', bandpower_edges=None, obs_type='1x2pt'):

    """
    Loop over numbers of bandpowers, and for each one, bin all blocks of unbinned covariance matrix and combine into
    a single matrix, saved to disk.

    Args:
        n_zbin (int): Number of redshift bins, assuming 1 position field and 1 shear field per redshift bin.
        lmin_in (int): Minimum l in the unbinned covariance.
        lmin_out (int): Minimum l to include in the binned covariance.
        lmax (int): Maximum l in the unbinned and binned covariance.
        n_bp_min (int): Minimum number of bandpowers to loop over (inclusive).
        n_bp_max (int): Maximum number of bandpowers to loop over (inclusive).
        input_filemask (str): Path to unbinned covariance blocks output by ``get_3x2pt_cov``, with ``{spec1_idx}`` and
                              ``{spec2_idx}`` placeholders.
        save_filemask (str): Path to save each binned covariance matrix, with ``{n_bp}`` placeholder.
    """

    # Calculate number of spectra and ells
    if obs_type == '3x2pt':
        n_fields = 2 * n_zbin
    else:
        assert obs_type == '1x2pt'
        n_fields = n_zbin
    n_spec = n_fields * (n_fields + 1) // 2
    n_ell_in = lmax - lmin_in + 1
    n_ell_out = lmax - lmin_out + 1

    # Loop over all numbers of bandpowers
    #for n_bp in range(n_bp_min, n_bp_max + 1):
    for bp_count, n_bp in enumerate(range(n_bp_min, n_bp_max + 1)):

        print(f'Starting n_bp = {n_bp} at {time.strftime("%c")}')

        print(f'Calculating binning matrix at {time.strftime("%c")}')
        #pbl = gaussian_cl_likelihood.python.simulation.get_binning_matrix(n_bp, lmin_out, lmax)

        if bandpower_edges is None:
            assert bandpower_spacing == 'log' or bandpower_spacing == 'lin'
            pbl = gaussian_cl_likelihood.python.simulation.get_binning_matrix(n_bp, lmin_out, lmax,
                                                                            bp_spacing=bandpower_spacing)
        if bandpower_edges is not None:
            assert bandpower_spacing == 'custom'
            pbl = gaussian_cl_likelihood.python.simulation.get_binning_matrix(n_bp, lmin_out, lmax,
                                                                              bp_spacing=bandpower_spacing,
                                                                              bp_edges=bandpower_edges[bp_count])

        assert pbl.shape == (n_bp, n_ell_out)

        print(f'Preallocating full covariance at {time.strftime("%c")}')
        n_data = n_spec * n_bp
        cov = np.full((n_data, n_data), np.nan)

        # Loop over all blocks
        for spec1 in range(n_spec):
            for spec2 in range(spec1 + 1):
                print(f'spec1 = {spec1}, spec2 = {spec2} at {time.strftime("%c")}')

                print('Loading block')
                block_path = input_filemask.format(spec1_idx=spec1, spec2_idx=spec2)
                with np.load(block_path) as data:
                    assert data['spec1_idx'] == spec1
                    assert data['spec2_idx'] == spec2
                    block_unbinned = data['cov_block']
                assert np.all(np.isfinite(block_unbinned))
                assert block_unbinned.shape == (n_ell_in, n_ell_in)
                lowl_skip = lmin_out - lmin_in
                block_unbinned = block_unbinned[lowl_skip:, lowl_skip:]
                assert block_unbinned.shape == (n_ell_out, n_ell_out)

                print('Binning block')
                block_binned = pbl @ block_unbinned @ pbl.T
                assert np.all(np.isfinite(block_binned))
                assert block_binned.shape == (n_bp, n_bp)

                print('Inserting block')
                cov[(spec1 * n_bp):((spec1 + 1) * n_bp), (spec2 * n_bp):((spec2 + 1) * n_bp)] = block_binned

        # Reflect to fill remaining elements, and check symmetric
        cov = np.where(np.isnan(cov), cov.T, cov)
        assert np.all(np.isfinite(cov))
        assert np.allclose(cov, cov.T, atol=0)

        # Save to disk
        save_path = save_filemask.format(n_bp=n_bp)
        header = (f'Full binned covariance matrix. Output from {__file__}.bin_combine_cov for n_bp = {n_bp}, '
                  f'n_zbin = {n_zbin}, lmin_in = {lmin_in}, lmin_out = {lmin_out}, lmax = {lmax}, '
                  f'input_filemask = {input_filemask}, at {time.strftime("%c")}')
        np.savez_compressed(save_path, cov=cov, n_bp=n_bp, header=header)
        print(f'Saved {save_path} at {time.strftime("%c")}')
        print()

    print(f'Done at {time.strftime("%c")}')


def combine_signal_noise_cov_spec(n_zbin, field, signal_cov_path, noise_cov_path, combined_cov_save_path):

    # Generate list of fields
    print('Generating list of fields')
    new_pos_field = lambda zbin: Field(field_type=FieldType.POSITION, zbin=zbin)
    new_she_field = lambda zbin: Field(field_type=FieldType.SHEAR, zbin=zbin)

    if field == 'E':
        fields = [new_she_field(zbin) for zbin in range(1, n_zbin + 1)]
        print(fields)

    else:
        assert field == 'N'
        fields = [new_pos_field(zbin) for zbin in range(1, n_zbin + 1)]
        print(fields)

    #fields = [new_field(zbin) for zbin in range(1, n_zbin + 1) for new_field in (new_pos_field, new_she_field)]
    #print(fields)
    # Generate list of target spectra (the ones we want the covariance for) in the correct (diagonal) order
    print('Generating list of spectra')
    n_field = len(fields)
    spectra = [PowerSpectrum(fields[row], fields[row + diag])
               for diag in range(n_field) for row in range(n_field - diag)]
    for i in range(len(spectra)):
        for j in range(len(spectra)):
            if i>=j:
                fname='cov_spec1_{}_spec2_{}.npz'.format(i,j)
                signal_cov_name = signal_cov_path+fname
                noise_cov_name = noise_cov_path+fname
                with np.load(signal_cov_name) as signal_cov_block_data:
                    signal_cov_block = signal_cov_block_data['cov_block']
                with np.load(noise_cov_name) as noise_cov_block_data:
                    noise_cov_block = noise_cov_block_data['cov_block']
                combined_block = signal_cov_block + noise_cov_block
                save_path = combined_cov_save_path+fname
                np.savez_compressed(save_path,cov_block=combined_block,spec1_idx=i,spec2_idx=j)
                print(f'Saved {save_path} at {time.strftime("%c")}')
                print()


def combine_signal_noise_cov(n_zbin, lmin_in, lmin_out, lmax, n_bp_min, n_bp_max, signal_cov_path, noise_cov_path,
                             combined_cov_save_path):
    '''
    Saves combined signal and noise covariance to disk
    Args:
        signal_cov_path: Path to signal covariance as calculated by get_3x2pt_cov_signal_pcl #Probbaly needs to be formatted. Have fun future Jonathan. Lol.
        noise_cov_path: Path to noise covariance as calculated by get_3x2pt_cov_noise
        combined_cov_save_path: Path to save combined signal + noise covariance
    Returns:

    '''

    for n_bp in range(n_bp_min, n_bp_max + 1):
        with np.load(signal_cov_path.format(n_bp=n_bp)) as signal_cov_data:
            signal_cov = signal_cov_data['cov']

        with np.load(noise_cov_path.format(n_bp=n_bp)) as noise_cov_data:
            noise_cov = noise_cov_data['cov']

        combined_cov = signal_cov + noise_cov
        # Save to disk
        save_path = combined_cov_save_path.format(n_bp=n_bp)
        header = (f'Full binned covariance matrix (Signal + Noise). Output from {__file__}.bin_combine_cov for n_bp = {n_bp}, '
                  f'n_zbin = {n_zbin}, lmin_in = {lmin_in}, lmin_out = {lmin_out}, lmax = {lmax}, '
                  f'signal covariance path = {signal_cov_path.format(n_bp=n_bp)}, '
                  f'noise covariance path = {noise_cov_path.format(n_bp=n_bp)}, at {time.strftime("%c")}')
        np.savez_compressed(save_path, cov=combined_cov, n_bp=n_bp, header=header)
        print(f'Saved {save_path} at {time.strftime("%c")}')
        print()

