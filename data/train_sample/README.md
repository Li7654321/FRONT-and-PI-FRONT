# Training Data Sample
The synthetic training dataset was generated using [MudPy](https://github.com/UO-Geophysics/MudPy), including its FakeQuakes workflow. A sample of the dataset is available as an attachment to the GitHub Release.
[Download Sample Dataset](https://github.com/Li7654321/FRONT-and-PI-FRONT/releases/tag/v3.0.0/longmenshan01.zip)
## Simulation parameters
| Parameter | Value |
| --- | --- |
| project_name| longmenshan01 |
| ncpus | 20 |
| target_Mw | 6.8–8.0, step 0.1 |
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
## Citation
If using these examples, cite the [MudPy source repository](https://github.com/UO-Geophysics/MudPy) and the exact MudPy version used for the simulations. A citable archived release is [MudPy v1.4, DOI: 10.5281/zenodo.15547301](https://doi.org/10.5281/zenodo.15547301); use this DOI only if v1.4 is the version actually used. Also cite the accompanying FRONT/PI-FRONT paper and the dataset release once those identifiers are available.
