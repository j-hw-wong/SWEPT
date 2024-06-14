"""
Functions to do with masks and mixing matrices.
"""

import copy
import time
import warnings

import healpy as hp
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import pymaster as nmt


def generate_mask(wmap_mask_path, nside, target_fsky, mask_save_path):
    """
    Generate a Stage-IV-like mask by manipulating the WMAP temperature mask and then adding random holes until the
    target sky fraction is reached.

    Args:
        wmap_mask_path (str): Path to the WMAP temperature mask fits file.
        nside (int): HEALPix resolution to use.
        target_fsky (float): Sky fraction to achieve. Holes will be added at random to reach this value.
        msk_save_path (str): Path to save the final mask as a fits file.
    """

    print('Loading')
    #input_mask = hp.fitsfunc.read_map(wmap_mask_path, dtype=float, verbose=False)
    #input_mask = hp.fitsfunc.read_map(wmap_mask_path, dtype=float)
    input_mask = hp.pixelfunc.ud_grade(hp.read_map(wmap_mask_path, dtype=float), nside)
    assert input_mask.shape == (hp.pixelfunc.nside2npix(nside), )
    assert np.amin(input_mask) == 0
    assert np.amax(input_mask) == 1
    print('Input mask fsky =', np.mean(input_mask))

    print('Rotating')
    rotated_mask = hp.rotator.Rotator(coord=['E', 'G']).rotate_map_alms(input_mask)
    print('Rotated mask fsky =', np.mean(rotated_mask))

    # Clip back to 0-1
    rotated_mask = np.clip(rotated_mask, 0, 1)
    assert np.amin(rotated_mask) == 0
    assert np.amax(rotated_mask) == 1

    print('Multiplying')
    dual_mask = input_mask * rotated_mask
    assert np.amin(dual_mask) == 0
    assert np.amax(dual_mask) == 1
    print('Dual mask fsky =', np.mean(dual_mask))

    # Iteratively take out holes until the desired fsky is reached
    mask = dual_mask
    rng = np.random.default_rng()
    npix = hp.pixelfunc.nside2npix(nside)
    while np.mean(mask) > target_fsky:

        # Select non-zero pixel as the centre of the hole
        have_hole_centre = False
        while not have_hole_centre:
            hole_centre = rng.integers(low=0, high=npix)
            #print(hole_centre)
            #print(hole_centre)
            if mask[hole_centre] > 0:
                have_hole_centre = True

        # Mask the centre
        mask[hole_centre] = 0

        # Mask the immediate neighbours, then their neighbours with a 50% chance, etc.
        neighbours = hole_centre
        hole_size = 0
        while hole_size < 6: # max size
            hole_size += 1
            neighbours = hp.pixelfunc.get_all_neighbours(nside, neighbours)

            if any(p == -1 for p in neighbours.flatten()):
                break
                #print('WARNING! Code may fail!!!')
                #print(im_neighbours.shape)
                #ids = np.where(im_neighbours==-1)
                #ids = np.flip(ids)
                #for id in ids:
                #    im_neighbours = np.delete(im_neighbours, id)
                #print(im_neighbours.shape)

            mask[neighbours] = 0
            #neighbours = np.copy(im_neighbours)

            if rng.integers(2) > 0:
                break
        print('fsky = {:.5f}'.format(np.mean(mask)), end='\r')
    print()

    # Final checks
    assert np.all(np.isfinite(mask))
    assert np.amin(mask) == 0
    assert np.amax(mask) == 1

    # Plot
    with warnings.catch_warnings():
        warnings.simplefilter('ignore') # ignore mollview warnings
        hp.visufunc.mollview(mask)
        plt.show()

    # Save to disk
    hp.fitsfunc.write_map(mask_save_path, mask, dtype=float)
    print('Saved ' + mask_save_path)


def plot_mask(mask_path, save_path=None):
    """
    Plot Mollweide projection of a mask, with colour bar.

    Args:
        mask_path (str): Path to mask FITS file.
        save_path (str, optional): Path to save plot to (default None). If None, plot is displayed.
    """

    # Load mask
    #mask = hp.fitsfunc.read_map(mask_path, dtype=float, verbose=False)
    mask = hp.fitsfunc.read_map(mask_path, dtype=float)

    # Calculate Mollweide projection
    with warnings.catch_warnings():
        warnings.simplefilter('ignore') # mollview's plotting code creates warnings
        mask_proj = hp.visufunc.mollview(mask, return_projected_map=True)
        plt.close()

    # Plot
    plt.rcParams.update({'font.size': 7})
    cmap = copy.copy(matplotlib.cm.get_cmap('cividis'))
    cmap.set_bad(color='white')
    plt.imshow(mask_proj, origin='lower', cmap=cmap, interpolation='none')
    plt.gca().axis('off')
    plt.colorbar(shrink=0.4, aspect=10)

    # Save or show
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('Saved ' + save_path)
    else:
        plt.show()


def get_3x2pt_mixmats(mask_path, nside, lmin, lmax_mix, lmax_out, input_lmax, save_path):
    """
    Calculate all 3x2pt mixing matrices from a mask using NaMaster, and save to disk in a single file.

    Args:
        mask_path (str): Path to mask FITS file. If None, full sky is assumed, in which case the mixing matrices should
                         be diagonal.
        nside (int): HEALPix resolution to use.
        lmin (int): Minimum l to include in mixing matrices.
        lmax_mix (int): Maximum l to include in input to mixing matrices.
        lmax_out (int): Maximum l to include in output from mixing matrices.
        save_path (str): Path to save output, as a single numpy .npz file containing all mixing matrices.
    """

    # Load and rescale mask, and calculate fsky
    if mask_path is not None:
        print('Loading and rescaling mask')
        print(nside)
        mask = hp.pixelfunc.ud_grade(hp.read_map(mask_path, dtype=float), nside)
        print(mask)
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
    bins = nmt.NmtBin.from_lmax_linear(input_lmax, 1)

    # Calculate mixing matrices for spin 0-0, 0-2 (equivalent to 2-0), and 2-2
    field_spin0 = nmt.NmtField(mask, None, spin=0, lite=True, lmax_sht=input_lmax)
    field_spin2 = nmt.NmtField(mask, None, spin=2, lite=True, lmax_sht=input_lmax)
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

    # Extract the relevant mixing matrices
    print('Extracting mixing matrices')
    # For 0-0 there is only a single mixing matrix
    mixmats_spin00 = workspace_spin00.get_coupling_matrix()
    mixmat_nn_to_nn = mixmats_spin00
    #print(mixmat_nn_to_nn)
    #print(mixmat_nn_to_nn.shape)
    # For 0-2 they are arranged NE->NE, NB->NE // NE->NB NB->NB, per l, so select every other row and column
    mixmats_spin02 = workspace_spin02.get_coupling_matrix()
    mixmat_ne_to_ne = mixmats_spin02[::2, ::2]
    # For 2-2 there are 4x4 elements per l, ordered EE, EB, BE, BB. We only need EE->EE and BB->EE,
    # so select every 4th row and the 1st and 4th columns from each block
    mixmats_spin22 = workspace_spin22.get_coupling_matrix()
    mixmat_ee_to_ee = mixmats_spin22[::4, ::4]
    mixmat_bb_to_ee = mixmats_spin22[::4, 3::4]

    # Check everything has the correct shape
    mixmat_shape = (input_lmax + 1, input_lmax + 1)
    assert mixmat_nn_to_nn.shape == mixmat_shape
    assert mixmat_ne_to_ne.shape == mixmat_shape
    assert mixmat_ee_to_ee.shape == mixmat_shape
    assert mixmat_bb_to_ee.shape == mixmat_shape

    # Trim to required output range
    mixmat_nn_to_nn = mixmat_nn_to_nn[lmin:(lmax_out + 1), lmin:(input_lmax+1)]
    mixmat_ne_to_ne = mixmat_ne_to_ne[lmin:(lmax_out + 1), lmin:(input_lmax+1)]
    mixmat_ee_to_ee = mixmat_ee_to_ee[lmin:(lmax_out + 1), lmin:(input_lmax+1)]
    mixmat_bb_to_ee = mixmat_bb_to_ee[lmin:(lmax_out + 1), lmin:(input_lmax+1)]
    print(mixmat_nn_to_nn)
    # Do some final checks
    n_ell_out = lmax_out - lmin + 1
    n_ell_in = input_lmax - lmin + 1
    mixmat_out_shape = (n_ell_out, n_ell_in)
    assert mixmat_nn_to_nn.shape == mixmat_out_shape
    assert mixmat_ne_to_ne.shape == mixmat_out_shape
    assert mixmat_ee_to_ee.shape == mixmat_out_shape
    assert mixmat_bb_to_ee.shape == mixmat_out_shape
    assert np.all(np.isfinite(mixmat_nn_to_nn))
    assert np.all(np.isfinite(mixmat_ne_to_ne))
    assert np.all(np.isfinite(mixmat_ee_to_ee))
    assert np.all(np.isfinite(mixmat_bb_to_ee))

    # Save to disk
    header = (f'Mixing matrices. Output from {__file__}.get_3x2pt_mixmats for mask_path = {mask_path}, '
              f'nside = {nside}, lmin = {lmin}, lmax_mix = {lmax_mix}, lmax_out = {lmax_out}, at {time.strftime("%c")}')
    np.savez_compressed(save_path, mixmat_nn_to_nn=mixmat_nn_to_nn, mixmat_ne_to_ne=mixmat_ne_to_ne,
                        mixmat_ee_to_ee=mixmat_ee_to_ee, mixmat_bb_to_ee=mixmat_bb_to_ee, header=header)
    print('Saved ' + save_path)
