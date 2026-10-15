# Training Data Sample
The synthetic training dataset was generated using [MudPy](https://github.com/UO-Geophysics/MudPy), including its FakeQuakes workflow. A sample of the dataset is available as an attachment to the GitHub Release.
[Download Sample Dataset](https://github.com/Li7654321/FRONT-and-PI-FRONT/releases/tag/v3.0.0/longmenshan01.zip)
## Simulation parameters
| Parameter | Value |
| --- | --- |
| project_name| longmenshan01 |
| ncpus | 20 |
| target_Mw | 6.8–8.3, step 0.1 |
| UTM_zone | 48N |
| time_epi | 2008-05-12T06:28:04Z |
| hypocenter | None |
| rake | 120°-160°, step 5° |
| scaling_law | T |
| hurst | 0.4-0.6 |
| max_slip | 20 m |
| max_slip_rule | True |
| rise_time | MH2017 |
| rise_time_depths | [10,15] |
| shear_wave_fraction_deep | 0.8 |
| source_time_function | dreger |
| stf_falloff_rate | 4 |
| num_modes | 72 |
| dk, pmin, pmax, kmax | 0.1, 0, 1, 20 |
| dt, NFFT | 0.5s, 512 |

## Velocity Models for the Longmenshan Region
The velocity models for the Longmenshan region are divided into two types: the **mountain model** and the **basin model**. When using Mudpy for forward modeling of ground motion, stations in the mountain region and stations in the basin region must be distinguished. Therefore, **two separate station waveform syntheses are required for a single earthquake**:
1. **Mountain region synthesis**: Use the mountain velocity model to synthesize waveforms for stations in the mountain region.
2. **Basin region synthesis**: Use the basin velocity model to synthesize waveforms for stations in the basin region.
Finally, the two sets of synthetic results are merged by station to obtain the complete records.
#### Mountain Velocity Model

| Layer | Thickness (km) | Vs (km/s) | Vp (km/s) | Density (g/cm³) |
|-------|----------------|-----------|-----------|-----------------|
| 1     | 2.5000         | 3.0926    | 5.2042    | 2.4600          |
| 2     | 2.5000         | 3.2556    | 5.3726    | 2.6000          |
| 3     | 15.0000        | 3.5062    | 5.9059    | 2.7500          |
| 4     | 15.0000        | 3.5023    | 6.0494    | 2.8500          |
| 5     | 15.0000        | 3.9588    | 6.8209    | 2.8200          |
| 6     | 20.0000        | 4.2377    | 7.7170    | 2.9300          |
| 7     | 0.0000         | 4.6776    | 8.0220    | 3.3000          |

#### Basin Velocity Model

| Layer | Thickness (km) | Vs (km/s) | Vp (km/s) | Density (g/cm³) |
|-------|----------------|-----------|-----------|-----------------|
| 1     | 2.5000         | 2.8200    | 5.3000    | 2.4000          |
| 2     | 2.5000         | 2.8900    | 5.4600    | 2.4600          |
| 3     | 15.0000        | 3.7000    | 6.2300    | 2.6500          |
| 4     | 15.0000        | 4.0000    | 6.7500    | 2.7500          |
| 5     | 15.0000        | 4.3900    | 7.4600    | 2.8200          |
| 6     | 0.0000         | 4.6800    | 8.0200    | 3.3000          |

## Velocity Models for the Maduo Region

| Layer | Thickness (km) | Vs (km/s) | Vp (km/s) | Density (g/cm³) |
|-------|----------------|-----------|-----------|-----------------|
| 1     | 2.00           | 3.55      | 6.05      | 2.65            |
| 2     | 8.00           | 3.74      | 6.25      | 2.72            |
| 3     | 20.00          | 3.80      | 6.35      | 2.78            |
| 4     | 20.00          | 3.90      | 6.40      | 2.85            |
| 5     | 18.00          | 4.05      | 7.10      | 3.00            |
| 6     | 0.00           | 4.70      | 8.00      | 3.30            |
## Citation
If using these examples, cite the [MudPy source repository](https://github.com/UO-Geophysics/MudPy) and the exact MudPy version used for the simulations. A citable archived release is [MudPy v1.4, DOI: 10.5281/zenodo.15547301](https://doi.org/10.5281/zenodo.15547301); use this DOI only if v1.4 is the version actually used. Also cite the accompanying FRONT/PI-FRONT paper and the dataset release once those identifiers are available.
