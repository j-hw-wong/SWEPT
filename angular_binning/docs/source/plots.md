Steps to produce all plots
==========================

Below are the steps to produce all figures in Chapter 5 of my PhD thesis, "Dependence of cosmological parameter constraints on angular binning of weak lensing two-point statistics".

## Figure 1: Mask

a) Generate custom Stage-IV-like mask from the WMAP temperature mask, using `mask.generate_mask`:

```python
[python]

wmap_mask_path = 'path-to-wmap-mask.fits.gz' # available from https://lambda.gsfc.nasa.gov/product/map/dr5/masks_get.cfm
nside = 512
target_fsky = 0.3
mask_save_path = 'path-to-save-custom-mask.fits.gz'

mask.generate_mask(wmap_mask_path, nside, target_fsky, mask_save_path)
```

b) Plot Mollweide projection of mask, using `mask.plot_mask`:

```python
[python]

mask_path = 'path-to-output-from-step-a.fits.gz'
save_path = 'path-to-save-figure.pdf'

mask.plot_mask(mask_path, save_path)
```

## Figure 2: w0–wa contour area against number of angular bins for full-sky power spectrum and correlation function

a) Produce CosmoSIS grid of w_0 and w_a – as [gaussian_cl_likelihood Fig. 4 step (a)](https://gaussian-cl-likelihood.readthedocs.io/en/latest/source/plots.html#figure-4-2d-posterior-with-discrepant-fiducial-parameters), except with w_0 ranging from -1.0119 to -0.9881 in 49 steps and w_a from -0.039 to 0.039 in 45 steps.

b) Produce a broader CosmoSIS grid for use with the low-n_bin correlation function – as [gaussian_cl_likelihood Fig. 1 step (a)](https://gaussian-cl-likelihood.readthedocs.io/en/latest/source/plots.html#figure-1-histograms-of-wishart-and-gaussian-1d-posterior-maxima-and-per-realisation-difference), except instead of using `gaussian_cl_likelihood.cosmosis_utils.generate_chain_input`, use `param_grids.reduced_grid_chains`:

```python
[python]

params = {
    'cosmological_parameters--w': {
        'min': -1.0218,
        'max': -0.9782,
        'steps': 49
    },
    'cosmological_parameters--wa': {
        'min': -0.072,
        'max': 0.072,
        'steps': 49
    }
}
upper_diag = [(-0.99, 0.072), (-0.9782, 0.02)]
lower_diag = [(-1.0218, -0.02), (-1.01, -0.072)]
n_chains = 11
output_dir = 'path-to-output-directory/' # must already exist

param_grids.reduced_grid_chains(params, upper_diag, lower_diag, n_chains, output_dir=None)
```

c) Generate noise power spectra and ells – as [gaussian_cl_likelihood Fig. 1 step (b)](https://gaussian-cl-likelihood.readthedocs.io/en/latest/source/plots.html#figure-1-histograms-of-wishart-and-gaussian-1d-posterior-maxima-and-per-realisation-difference).

d) Produce mock observed power spectra directly from fiducial power spectra with no realisation noise, using `loop_likelihood_nbin.obs_from_fid`:

```python
[python]

input_dir = 'path-to-fiducial-subdirectory-from-step-a-or-b/'
output_path = 'path-to-save-output.npz'
n_zbin = 5
lmax = 2000
lmin =  2

loop_likelihood_nbin.obs_from_fid(input_dir, output_path, n_zbin, lmax, lmin)
```

e) Run the full-sky bandpower likelihood for all numbers of bandpowers, using `loop_likelihood_nbin.like_bp_gauss_loop_nbin`:

```python
[python]

grid_dir = 'path-to-cosmosis-grid-from-step-a/'
n_bps = np.arange(1, 31)
n_zbin = 5
lmax = 2000
lmin_like = 10
lmin_in = 2
fid_base_dir = 'path-to-fiducial-subdirectory-from-step-a-or-b/'
fid_pos_pos_dir = fid_base_dir + 'galaxy_cl/'
fid_she_she_dir = fid_base_dir + 'shear_cl/'
fid_pos_she_dir = fid_base_dir + 'galaxy_shear_cl/'
pos_nl_path = 'path-to-pos_nl-from-step-c.txt'
she_nl_path = 'path-to-she_nl-from-step-c.txt'
noise_ell_path = 'path-to-noise_ell-from-step-c.txt'
pbl_save_dir = 'directory-to-save-pbl-matrices/'
obs_bp_save_dir = 'directory-to-save-obs_bp/'
inv_cov_save_dir = 'directory-to-save-inverse-covariance/'
varied_params = ['w', 'wa']
like_save_dir = 'directory-to-save-log-likelihood-files/'

loop_likelihood_nbin.like_bp_gauss_loop_nbin(grid_dir, n_bps, n_zbin, lmax, lmin_like, lmin_in, fid_pos_pos_dir, fid_she_she_dir, fid_pos_she_dir, pos_nl_path, she_nl_path, noise_ell_path, pbl_save_dir, obs_bp_save_dir, inv_cov_save_dir, varied_params, like_save_dir)
```

f) Run the full-sky correlation function likelihood for all numbers of theta bins, using `loop_likelihood_nbin.like_cf_gauss_loop_nbin`:

```python
[python]

# It is necessary to split this into two different grids
# because the performance deteriorates so much for low nbin

# Common parameters
n_zbin = 5
lmin = 10
lmax = 2000
theta_min = np.radians(0.1) # 0.1 deg
theta_max = np.radians(10)  # 10 deg
fid_base_dir = 'path-to-fiducial-subdirectory-from-step-a-or-b/'
fid_pos_pos_dir = fid_base_dir + 'galaxy_cl/'
fid_she_she_dir = fid_base_dir + 'shear_cl/'
fid_pos_she_dir = fid_base_dir + 'galaxy_shear_cl/'
obs_path = 'path-to-observation-from-step-d.npz'
survey_area_sqdeg = 15000
gals_per_sqarcmin_per_zbin = 30 / n_zbin
sigma_e = 0.3
varied_params = ['w', 'wa']
like_save_dir = 'directory-to-save-log-likelihood-files/'

# Low nbin run
grid_dir_lo = 'path-to-broad-cosmosis-grid-from-step-b/'
n_theta_bins_lo = np.arange(1, 6)
loop_likelihood_nbin.like_cf_gauss_loop_nbin(grid_dir_lo, n_theta_bins_lo, n_zbin, lmin, lmax, theta_min, theta_max, fid_pos_pos_dir, fid_she_she_dir, fid_pos_she_dir, obs_path, survey_area_sqdeg, gals_per_sqarcmin_per_zbin, sigma_e, varied_params, like_save_dir)

# High nbin run
grid_dir_hi = 'path-to-standard-cosmosis-grid-from-step-a/'
n_theta_bins_hi = np.arange(6, 31)
loop_likelihood_nbin.like_cf_gauss_loop_nbin(grid_dir_hi, n_theta_bins_hi, n_zbin, lmin, lmax, theta_min, theta_max, fid_pos_pos_dir, fid_she_she_dir, fid_pos_she_dir, obs_path, survey_area_sqdeg, gals_per_sqarcmin_per_zbin, sigma_e, varied_params, like_save_dir)
```

g) Plot w0–wa contour area against number of angular bins for the power spectrum and correlation function side-by-side, using `error_vs_nbin.area_vs_nbin`:

```python
[python]

cl_like_filemask = 'path-to-output-directory-from-step-e/like_lmax2000_{n_bp}bp.txt'
cf_like_filemask = 'path-to-output-directory-from-step-f/like_thetamin0.1_{n_bin}bins.txt'
contour_levels_sig = np.arange(1, 3, 100)
n_bps = np.arange(1, 31)
n_theta_bins = np.arange(1, 31)
save_path = 'path-to-save-figure.pdf' # or None to just display it

error_vs_nbin.area_vs_nbin(cl_like_filemask, cf_like_filemask, contour_levels_sig, n_bps, n_theta_bins, save_path)
```

## Figure 3: w0–wa posteriors from power spectrum with different numbers of bandpowers

a) As Fig. 2 steps (a), (c), (e), to produce likelihood files, except it is only necessary to evaluate `loop_likelihood_nbin.like_bp_gauss_loop_nbin` for the chosen numbers of bandpowers, so in step (e) use
```python
n_bps = [1, 5, 10, 30]
```

b) Plot the posteriors from the likelihood files, using `posterior.cl_posts:`

```python
[python]

log_like_filemask = 'path-to-output-directory-from-fig-2-step-e/like_lmax2000_{n_bp}bp.txt'
contour_levels_sig = [1, 2, 3]
n_bps = [[30, 1], [30, 5], [30, 10]]
colours = ['C0', 'C1']
linestyles = [['-'], [(0, (2, 2))]]
plot_save_path = 'path-to-save-figure.pdf' # or None to just display it

posterior.cl_posts(log_like_filemask, contour_levels_sig, n_bps, colours, linestyles, plot_save_path=plot_save_path)
```

## Figure 4: w0–wa posteriors from correlation function with different numbers of theta bins

a) As Fig. 2 steps (a), (d), (f) to produce likelihood files, except it is only necessary to evaluate `loop_likelihood_nbin.like_cf_gauss_loop_nbin` for the chosen numbers of theta bins, and the broader grid is not necessary here, so in step (f) use a single call to `like_cf_gauss_loop_nbin` with
```python
n_theta_bins = [5, 10, 20, 30]
```

b) Plot the posteriors from the likelihood files, using `posterior.cf_posts:`

```python
[python]

log_like_filemask = 'path-to-output-directory-from-step-f/like_thetamin0.1_{n_bin}bins.txt'
contour_levels_sig = [1, 2, 3]
n_bins = [[30, 5], [30, 10], [30, 20]]
colours = ['C0', 'C1']
linestyles = [['-'], [(0, (2, 2))]]
plot_save_path = 'path-to-save-figure.pdf' # or None to just display it

posterior.cf_posts(log_like_filemask, contour_levels_sig, n_bins, colours, linestyles, plot_save_path=plot_save_path)
```

## Figure 5: w0–wa contour area against number of angular bins for cut-sky power spectrum and correlation function

a) Produce a broad CosmoSIS grid of w_0 and w_a to lmax 5000 – as Fig. 2 step (b) (i.e. the same setup as the full-sky low-n_theta_bin grid), except in `gaussian_cl_likelihood/ini/tomo_3x2_pipeline.ini` use
```ini
ell_max = 5000.0
n_ell = 4999
```

b) Produce noise power spectra – as Fig. 2 step (c).

c) Produce custom Stage IV-like mask – as Fig. 1 step (a).

d) Produce 3x2pt mixing matrices for the mask, using `mask.get_3x2pt_mixmats`:

```python
[python]

mask_path = 'path-to-mask-from-step-c.fits.gz'
nside = 2048
lmin = 2
lmax_mix = 5000
lmax_out = 2000
save_path = 'path-to-save-mixmats.npz'

mask.get_3x2pt_mixmats(mask_path, nside, lmin, lmax_mix, lmax_out, save_path)
```

e) Calculate unbinned Gaussian covariance blocks, using `covariance.get_3x2pt_cov:

```python
[python]

n_zbin = 5
theory_cl_dir = 'path-to-fiducial-directory-from-step-a/'
pos_pos_filemask = theory_cl_dir + 'galaxy_cl/bin_{hi_zbin}_{lo_zbin}.txt'
pos_she_filemask = theory_cl_dir + 'galaxy_shear_cl/bin_{pos_zbin}_{she_zbin}.txt'
she_she_filemask = theory_cl_dir + 'shear_cl/bin_{hi_zbin}_{lo_zbin}.txt'
lmax_in = 5000
lmin_in = 2
lmax_out = 2000
lmin_out = 2
pos_nl_path = 'path-to-pos_nl-from-step-b.txt'
she_nl_path = 'path-to-she_nl-from-step-b.txt'
noise_lmin = 2
mask_path = 'path-to-mask-from-step-c.fits.gz'
nside = 2048
save_filemask = 'path-to-output-directory/cov_spec1_{spec1_idx}_spec2_{spec2_idx}.npz'

covariance.get_3x2pt_cov(n_zbin, pos_pos_filemask, pos_she_filemask, she_she_filemask, lmax_in, lmin_in, lmax_out, lmin_out, pos_nl_path, she_nl_path, noise_lmin, mask_path, nside, save_filemask)
```

f) Form binned covariance matrices from the unbinned blocks for all numbers of bandpowers, using `covariance.bin_combine_cov`:

```python
[python]

n_zbin = 5
lmin_in = 2
lmin_out = 10
lmax = 2000
n_bp_min = 1
n_bp_max = 30
input_filemask = 'path-to-output-directory-from-step-e/cov_spec1_{spec1_idx}_spec2_{spec2_idx}.npz'
save_filemask = 'path-to-output-directory/cov_{n_bp}bp.npz'

covariance.bin_combine_cov(n_zbin, lmin_in, lmin_out, lmax, n_bp_min, n_bp_max, input_filemask, save_filemask)
```

g) Run the cut-sky bandpower likelihood for all numbers of bandpowers, using `loop_likelihood_nbin.like_cf_gauss_loop_nbin`:

```python
[python]

grid_dir = 'path-to-cosmosis-grid-from-step-a/'
n_bps = np.arange(1, 31)
n_zbin = 5
lmax_like = 2000
lmin_like = 10
lmax_in = 5000
lmin_in = 2
fid_base_dir = 'path-to-fiducial-directory-from-step-a/'
fid_pos_pos_dir = fid_base_dir + 'galaxy_cl/'
fid_she_she_dir = fid_base_dir + 'shear_cl/'
fid_pos_she_dir = fid_base_dir + 'galaxy_shear_cl/'
pos_nl_path = 'path-to-pos_nl-from-step-b.txt'
she_nl_path = 'path-to-she_nl-from-step-b.txt'
mixmats_path = 'path-to-mixing-matrices-from-step-d.npz'
bp_cov_filemask = 'path-to-output-directory-from-step-f/cov_{n_bp}bp.npz'
binmixmat_save_dir = 'path-to-directory-to-save-binned-mixmats/'
obs_bp_save_dir = 'path-to-directory-to-save-observations/'
varied_params = ['w', 'wa']
like_save_dir = 'directory-to-save-log-likelihood-files/'

loop_likelihood_nbin.like_bp_gauss_mix_loop_nbin(grid_dir, n_bps, n_zbin, lmax_like, lmin_like, lmax_in, lmin_in, fid_pos_pos_dir, fid_she_she_dir, fid_pos_she_dir, pos_nl_path, she_nl_path, mixmats_path, bp_cov_filemask, binmixmat_save_dir, obs_bp_save_dir, varied_params, like_save_dir)
```

h) Run the cut-sky correlation function likelihood for all numbers of theta bins – as Fig. 2 step (f), except use the broad grid produced in step (a) for all numbers of theta bins, and set
```python
cov_fsky = 0.3
```

i) Plot w0–wa contour area against number of angular bins for the cut-sky power spectrum and correlation function side-by-side – as Fig. 2 step (g), except point to the log-likelihood files produced in steps (g) and (h).

## Figure 6: Grid of single-parameter error against number of angular bins, for full-sky power spectrum and correlation function, for different parameters

a) Produce all single-parameter CosmoSIS grids – repeat [gaussian_cl_likelihood Fig. 1 step (a)](https://gaussian-cl-likelihood.readthedocs.io/en/latest/source/plots.html#figure-1-histograms-of-wishart-and-gaussian-1d-posterior-maxima-and-per-realisation-difference) for each parameter and lmax combination, with the following changes:

1. Include a call to `sigma8_rescale` in `gaussian_cl_likelihood/ini/tomo_3x2_pipeline.ini`:
    ```ini
    modules = consistent_parameters camb sigma8_rescale halofit no_bias gaussian_window project_2d
    ```
    with a corresponding section
    ```ini
    ; Rescale sigma_8 in order to be able to sample over it
    [sigma8_rescale]
    file = cosmosis-standard-library/utility/sample_sigma8/sigma8_rescale.py
    ```

2. Use `prior.ini` in place of `de_grid.ini`.

3. Use the relevant `ell_max` and `n_ell` parameters for each grid in `tomo_3x2_pipeline.ini` (i.e. `n_ell` = 999, 1999, 4999 for `ell_max` = 1000, 2000, 5000, respectively, assuming `ell_min` = 2.

4. Instead of `gaussian_cl_likelihood.cosmosis_utils.generate_chain_input`, use `param_grids.single_param_chains` to produce chain input for each combination of parameter and lmax:

    ```python
    [python]

    fid_params = {
        'w': -1.0,
        'wa': 0.0,
        'omega_m': 0.314,
        'omega_b': 0.049,
        'sigma8_input': 0.826,
        'h0': 0.673,
        'n_s': 0.966,
    }
    steps = 101
    n_chains = 11 # assuming 12 CPUs leaving one spare
    output_base_dir = 'path-to-output-directory/'

    # Loop over all the combinations of parameter and lmax
    half_ranges = {
        'w': {
            'lmax1000': 0.01,
            'lmax2000': 0.0056,
            'lmax5000': 0.0056
        },
        'wa': {
            'lmax1000': 0.03,
            'lmax2000': 0.0204,
            'lmax5000': 0.0204
        },
        'omega_m': {
            'lmax1000': 0.02,
            'lmax2000': 0.012,
            'lmax5000': 0.012
        },
        'omega_b': {
            'lmax1000': 0.02,
            'lmax2000': 0.0026,
            'lmax5000': 0.0026
        },
        'sigma8_input': {
            'lmax1000': 0.01,
            'lmax2000': 0.001,
            'lmax5000': 0.001
        },
        'h0': {
            'lmax1000': 0.04,
            'lmax2000': 0.025,
            'lmax5000': 0.025
        },
        'n_s': {
            'lmax1000': 0.02,
            'lmax2000': 0.012,
            'lmax5000': 0.012
        }
    }
    for param_to_vary in fid_params:
        for lmax in [1000, 2000, 5000]:
            half_range = half_ranges[param_to_vary][f'lmax{lmax}']
            output_dir = output_base_dir + f'{param_to_vary}_lmax{lmax}/'

            param_grids.single_param_chains(fid_params, param_to_vary, half_range, steps, n_chains, output_dir)
    ```

b) Produce noise power spectra – as Fig. 2 step (c).

c) Produce mock observed power spectra identical to fiducial power spectra – as Fig. 2 step (d).

d) For each combination of parameter and lmax, evaluate the full-sky bandpower likelihood for all numbers of bandpowers – as Fig. 2 step (e).

e) For each combination of parameter and theta_min (where the respective corresponding theta_min to lmax 1000, 2000, 5000 are 0.2, 0.1, 0.03 deg), evaluate the full-sky correlation function likelihood for all numbers of theta bins – as Fig. 2 step (f).

f) Prepare the data for the grid of single-parameter error against number of angular bins, using `error_vs_nbin.prepare_width_vs_nbin_grid`:

```python
[python]

# These will depend on how exactly the file structure is laid out
cl_like_filemask = 'path-to-data/cl/{param}/like_lmax{lmax}_{n_bp}bp.txt'
cf_like_filemask = 'path-to-data/cf/{param}/like_thetamin{theta_min_deg}_{n_bin}bins.txt'

contour_levels_sig = np.arange(1, 3, 100)
n_bps = np.arange(1, 31)
n_theta_bins = np.arange(1, 31)
params = ['w', 'wa', 'omega_m', 'omega_b', 'sigma8_input', 'h0', 'n_s']
lmaxes = [1000, 2000, 5000]
theta_min_degs = [0.2, 0.1, 0.03]
data_save_path = 'path-to-save-data.npz'

error_vs_nbin.prepare_width_vs_nbin_grid(cl_like_filemask, cf_like_filemask, contour_levels_sig, n_bps, n_theta_bins, params, lmaxes, theta_min_degs, data_save_path)
```

g) Plot the data evaluated in step (f), using `error_vs_nbin.plot_width_vs_nbin_grid`:

```python
[python]

data_path = 'path-to-output-from-step-f.npz'
param_labels = {
    'w': 'w_0',
    'wa': 'w_a',
    'omega_m': r'\Omega_\mathrm{m}',
    'omega_b': r'\Omega_\mathrm{b}',
    'sigma8_input': r'\sigma_8',
    'h0': 'h',
    'n_s': r'n_\mathrm{s}'
}
plot_save_path = 'path-to-save-figure.pdf' # or None to display it

error_vs_nbin.plot_width_vs_nbin_grid(data_path, param_labels, plot_save_path)
```

## Figure 7: Single-parameter error against number of bandpowers for different lmax values, scaled as sqrt(lmax)

a) As Fig. 6 steps (a), (b), (d), but only for parameters `w`, `wa`, `omega_m` and `sigma8_input`.

b) Plot grid of single-parameter error against number of bandpowers adjusted as sqrt(lmax) for these four parameters, using `error_vs_nbin.width_vs_nbin_sqrt_lmax`:

```python
[python]

log_like_filemask = 'path-to-data/cl/{param}/like_lmax{lmax}_{n_bp}bp.txt' # Will depend on how exactly the file structure is laid out
contour_levels_sig = np.arange(1, 3, 100)
n_bps = np.arange(1, 31)
params = [['w', 'wa'], ['omega_m', 'sigma8_input']]
param_labels = {
    'w': 'w_0',
    'wa': 'w_a',
    'omega_m': r'\Omega_\mathrm{m}',
    'sigma8_input': r'\sigma_8'
}
lmaxes = [1000, 2000, 5000]
plot_save_path = 'path-to-save-figure.pdf' # or None to display it

error_vs_nbin.width_vs_nbin_sqrt_lmax(log_like_filemask, contour_levels_sig, n_bps, params, param_labels, lmaxes, plot_save_path)
```

## Figure 8: w_0–w_a  joint uncertainty against number of cut-sky bandpowers for fsky approximation compared to improved NKA

a) As Fig. 5 steps (a)–(g), to run cut-sky power spectrum likelihood using improved NKA covariance.

b) Produce likelihood files using the fsky approximation – as Fig. 2 step (e), except set
```python
cov_fsky = 0.3
```
in the call to `loop_likelihood_nbin.like_bp_gauss_loop_nbin`.

c) Calculate and plot w0–wa joint uncertainty against number of cut-sky bandpowers for both covariance methods, using `error_vs_nbin.area_vs_nbin_fsky_inka`:

```python
[python]

inka_like_filemask = 'path-to-output-from-step-a/like_lmaxlike2000_{n_bp}bp.txt'
fsky_like_filemask = 'path-to-output-from-step-b/like_lmax{lmax}_{n_bp}bp.txt'
contour_levels_sig = np.arange(1, 3, 100)
n_bps = np.arange(1, 31)
plot_save_path = 'path-to-save-figure.pdf' # or None to display it

error_vs_nbin.area_vs_nbin_fsky_inka(inka_like_filemask, fsky_like_filemask, contour_levels_sig, n_bps, plot_save_path)
```

## Figure 9: Shear power spectra at different points in the w_0–w_a plane

a) Generate CosmoSIS grid of w_0 and w_a to lmax 2000 – as Fig. 2 step (a).

b) Extract a list of parameter values diagonally across the grid, perpendicular to the degeneracy direction, using `param_grids.get_diagonal_params`:

```python
[python]

grid_dir = 'path-to-cosmosis-grid-from-step-a/'
varied_params = ['w', 'wa']
save_path = 'path-to-save-params.txt'

param_grids.get_diagonal_params(grid_dir, varied_params, save_path)
```

c) Compile the first (lowest redshift) shear auto-power spectrum for each point on the diagonal into a single file, using `param_grids.load_diagonal_shear_cl(grid_dir, diag_params_path, save_path)`:

```python
[python]

grid_dir = 'path-to-cosmosis-grid-from-step-a/'
diag_params_path = 'path-to-list-of-params-from-step-b.txt'
save_path = 'path-to-save-spectra.npz'

param_grids.load_diagonal_shear_cl(grid_dir, diag_params_path, save_path)
```

d) Produce shear noise power spectrum – as Fig. 2 step (c).

e) Plot all of the power spectra collected in step (c) along with their distance from the fiducial model, using `snr_per_bin.plot_cl_cf` (also produces the equivalent plot for the correlation function):

```python
[python]

diag_she_cl_path = 'path-to-power-spectra-from-step-c.npz'
she_nl_path = 'path-to-she_nl-from-step-d.txt'
lmin = 2
lmax = 2000
theta_min = np.radians(0.1)
theta_max = np.radians(10)
n_theta_bin = 30
survey_area_sqdeg = 15000
gals_per_sqarcmin = 6
sigma_e = 0.3
plot_save_dir = 'directory-to-save-plots/' # must already exist, or None to display

snr_per_bin.plot_cl_cf(diag_she_cl_path, she_nl_path, lmin, lmax, theta_min, theta_max, n_theta_bin, survey_area_sqdeg, gals_per_sqarcmin, sigma_e, plot_save_dir=plot_save_dir)
```

## Figure 10: Shear correlation functions at different points in the w_0–w_a plane

a) Identical to Fig. 9: the same process produces both plots.

## Figure 11: Shear correlation functions at different points in the w_0–w_a plane, compared between 30 and 15 theta bins

a) As Fig. 9 steps (a)–(c).

b) Plot shear correlation functions and distance from fiducial model for 30 and 15 theta bins side-by-side, using `snr_per_bin.plot_cf_nbin`:

```python
[python]

diag_she_cl_path = 'path-to-power-spectra-from-fig-9-step-c.npz'
lmin = 2
lmax = 2000
theta_min = np.radians(0.1)
theta_max = np.radians(10)
n_bin_1 = 30
n_bin_2 = 15
survey_area_sqdeg = 15000
gals_per_sqarcmin = 6
sigma_e = 0.3
plot_save_path = 'path-to-save-figure.pdf' # or None to display it

snr_per_bin.plot_cf_nbin(diag_she_cl_path, lmin, lmax, theta_min, theta_max, n_bin_1, n_bin_2, survey_area_sqdeg, gals_per_sqarcmin, sigma_e, plot_save_path=plot_save_path)
```

## Figure 12: Shear correlation functions at different points in the w_0–w_a plane, compared between 2 and 1 theta bins

a) As Fig. 11, except with 2 and 1 theta bins in the call to `snr_per_bin.plot_cf_nbin`:
```python
n_bin_1 = 2
n_bin_2 = 1
```

## Figure 13: Validation of the covariance-weighted distance statistic

a) As Fig. 9 steps (a)–(b), to generate CosmoSIS grid of w_0 and w_a and compile a list of parameter values across the diagonal perpendicular to the degeneracy direction.

b) Compile all 3x2pt power spectra (excluding B-modes) for each point along the diagonal into a single file, using `param_grids.load_diagonal_3x2pt_cl`:

```python
[python]

grid_dir = 'path-to-cosmosis-grid-from-fig-9-step-a/'
diag_params_path = 'path-to-diagonal-parameter-values-from-fig-9-step-b.txt'
save_path = 'path-to-save-all-power-spectra.npz'
lmax = 2000
lmin_in = 2
n_zbin = 5

param_grids.load_diagonal_3x2pt_cl(grid_dir, diag_params_path, save_path, lmax, lmin_in, n_zbin)
```

c) Compile all 3x2pt power spectra including zero B-modes for each point along the diagonal into a single file, using `param_grids.load_diagonal_3x2pt_cl_with_b`:

```python
[python]

grid_dir = 'path-to-cosmosis-grid-from-fig-9-step-a/'
diag_params_path = 'path-to-diagonal-parameter-values-from-fig-9-step-b.txt'
save_path = 'path-to-save-all-power-spectra-with-b.npz'
lmax = 2000
lmin_in = 2
n_zbin = 5

param_grids.load_diagonal_3x2pt_cl_with_b(grid_dir, diag_params_path, save_path, lmax, lmin_in, n_zbin)
```

d) Produce noise power spectra – as Fig. 2 step (c).

e) Calculate covariance-weighted distances and save to file for fast plotting, using `cov_distance.prepare_validation`:

```python
[python]

n_zbin = 5
lmax = 2000
lmin = 2
diag_cls_no_b_path = 'path-to-power-spectra-from-step-b.npz'
diag_cls_with_b_path = 'path-to-power-spectra-with-b-modes-from-step-c.npz'
pos_nl_path = 'path-to-pos_nl-from-step-d.txt'
she_nl_path = 'path-to-she_nl-from-step-d.txt'
theta_min = np.radians(0.1)
theta_max = np.radians(60) # For consistency with
n_theta_bin = 10           # the posterior measurements
survey_area_sqdeg = 15000
gals_per_sqarcmin_per_zbin = 30 / n_zbin
sigma_e = 0.3
data_save_path = 'path-to-save-output.npz'

cov_distance.prepare_validation(n_zbin, lmax, lmin, diag_cls_no_b_path, diag_cls_with_b_path, pos_nl_path, she_nl_path, theta_min, theta_max, n_theta_bin, survey_area_sqdeg, gals_per_sqarcmin_per_zbin, sigma_e, data_save_path)
```

f) Produce the validation plot of covariance-weighted distance against posterior-measured distance for the power spectrum and correlation function side-by-side, using `cov_distance.plot_validation`:

```python
[python]

plot_data_path = 'path-to-output-from-step-e.npz'
plot_save_path = 'path-to-save-figure.pdf' # or None to display it

cov_distance.plot_validation(plot_data_path, plot_save_path)
```

## Figure 14: Covariance-weighted distance against number of angular bins for power spectrum with three different weightings and correlation function

a) As Fig. 13 steps (a)–(d).

b) Calculate covariance-weighted distance for the different configurations and save to disk for fast plotting, using `cov_distance.prepare_dist_vs_nbin`:

```python
[python]

n_zbin = 5
lmax = 2000
lmin = 10
diag_cls_no_b_path = 'path-to-power-spectra-from-fig-13-step-b.npz'
diag_cls_with_b_path = 'path-to-power-spectra-with-b-modes-from-fig-13-step-c.npz'
n_bps = np.arange(1, 31)
pos_nl_path = 'path-to-pos_nl-from-fig-13-step-d.txt'
she_nl_path = 'path-to-she_nl-from-fig-13-step-d.txt'
theta_min = np.radians(0.1)
theta_max = np.radians(10)
n_theta_bins = np.arange(1, 31)
survey_area_sqdeg = 15000
gals_per_sqarcmin_per_zbin = 30 / n_zbin
sigma_e = 0.3
data_save_path = 'path-to-save-output.npz'

cov_distance.prepare_dist_vs_nbin(n_zbin, lmax, lmin, diag_cls_no_b_path, diag_cls_with_b_path, n_bps, pos_nl_path, she_nl_path, theta_min, theta_max, n_theta_bins, survey_area_sqdeg, gals_per_sqarcmin_per_zbin, sigma_e, data_save_path)
```

c) Plot covariance weighted distance against number of angular bins for the power spectrum with the three different weightings and the correlation function, all in a single figure, using `cov_distance.plot_dist_vs_nbin`:

```python
[python]

plot_data_path = 'path-to-output-from-step-b.npz'
plot_save_path = 'path-to-save-figure.pdf' # or None to display it

cov_distance.plot_dist_vs_nbin(plot_data_path, plot_save_path)
```

## Figure 15: Error on w_0 as function of number of angular bins for different noise levels

a) Produce single-parameter CosmoSIS grids of w_0 – as Fig. 6 step (a) for the following `half_range` values:

1. For baseline noise, x0.5 noise and x0.01 noise,
    ```python
    half_range = 0.0056
    ```

2. For x2 noise,
    ```python
    half_range = 0.01
    ```

3. For x100 noise,
    ```python
    half_range = 0.2
    ```

b) Produce noise power spectra for the five different noise levels – as Fig. 2 step (c), adjusting noise levels as appropriate (e.g. to double the noise level, multiply `gals_per_sq_arcmin` by 0.5).

c) Produce mock observed power spectra identical to fiducial power spectra – as Fig. 2 step (d). (No change is required for the different noise levels, because this is only used for the correlation function, for which noise only enters via the covariance.)

d) For each power spectrum noise level (baseline, x2, x100, x0.5, x0.01), repeat Fig. 2 step (e) to produce likelihood files for all numbers of bandpowers. Adjust the paths of the noise power spectra and CosmoSIS grids as appropriate in the call to `loop_likelihood_nbin.like_bp_gauss_loop_nbin`:

```python
grid_dir = 'path-to-cosmosis-grid-from-step-a-for-this-noise-level/'
pos_nl_path = 'path-to-pos_nl-from-step-b-for-this-noise_level.txt'
she_nl_path = 'path-to-she_nl-from-step-b-for-this-noise_level.txt'
like_save_dir = 'path-to-save-likelihood-files-for-this-noise-level/'
```

e) For each correlation function noise level (baseline, x2, x0.5), repeat Fig. 2 step (f) to produce likelihood files for all numbers of theta bins. Adjust the noise levels and path to CosmoSIS grid as appropriate in the call to `loop_likelihood_nbin.like_cf_gauss_loop_nbin`:

```python
grid_dir = 'path-to-cosmosis-grid-from-step-a-for-this-noise-level/'
noise_multiplier = 2 # or 1, or 0.5
gals_per_sqarcmin_per_zbin = 30 / n_zbin / noise_multiplier
like_save_dir = 'path-to-save-likelihood-files-for-this-noise-level/'
```

f) Plot error on w_0 against number of angular bins for the power spectrum with two different ranges of noise and the correlation function, using `error_vs_nbin.width_vs_nbin_noise`:

```python
[python]

# These will depend on how the file structure is laid out
cl_like_filemask = 'path-to-cl-likelihood-files-from-step-d/x{noise_level}noise/like_lmax2000_{n_bp}bp.txt'
cf_like_filemask = 'path-to-cf-likelihood-files-from-step-e/x{noise_level}noise/like_thetamin0.1_{n_bin}bins.txt'

contour_levels_sig = np.arange(1, 3, 100)
n_bps = np.arange(1, 31)
n_theta_bins = np.arange(1, 31)
plot_save_path = 'path-to-save-figure.pdf' # or None to display it

error_vs_nbin.width_vs_nbin_noise(cl_like_filemask, cf_like_filemask, contour_levels_sig, n_bps, n_theta_bins, plot_save_path)
```

## Figure 16: Power spectrum per-l distance from fiducial model with x100 noise

a) Produce shear noise power spectrum with x100 noise – as Fig. 15 step (b).

b) As Fig. 9, except use the x100 noise power spectrum and apply a factor 1/100 to the galaxy number density in the call to `snr_per_bin.plot_cl_cf` (this also produces the equivalent plot for the correlation function):

```python
she_nl_path = 'path-to-x100-she_nl-from-step-a.txt'
gals_per_sqarcmin = 6 / 100
```

## Figure 17: Correlation function per-bin distance from fiducial model with x100 noise

a) Identical to Fig. 16: the same process produces both plots.
