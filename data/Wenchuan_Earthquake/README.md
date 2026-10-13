# 2008 Wenchuan Earthquake Test Data
This directory contains three-component displacement waveforms for testing FRONT and PI-FRONT on the 2008 Wenchuan earthquake. The displacement records were obtained by integrating strong-motion accelerometer data after baseline correction following the VMD-based method of [Zhou et al. (2026)](https://doi.org/10.1016/j.soildyn.2026.110089). The processed displacement waveforms were then low-pass filtered at **0.5 Hz** for use as model inputs.
## Acceleration-to-displacement processing
The permanent-displacement recovery follows the method described by Zhou et al. (2026):

1. Subtract the pre-event mean from each acceleration component and integrate to obtain an initial velocity history.
2. Apply variational mode decomposition (VMD) to the velocity history. Use its long-period residual to guide a three-segment baseline fit rather than removing the residual directly.
3. Subtract the fitted baseline from the velocity history and integrate the corrected velocity to obtain displacement. The first and third segments are selected through a search on the VMD residual; a cubic fit joins them in the middle segment.
4. Estimate each component's permanent displacement as the mean of the final **10 s** of its corrected displacement history.
5. Low-pass filter the processed displacement records at **0.5 Hz** for the model input used here.

The cited paper reports a VMD penalty factor of **5000** and **10 modes** for its analyses. Those are the published method settings; the exact implementation settings used to create every file in this directory should be recorded with the processing script or data provenance when available. The 0.5 Hz low-pass cutoff describes the model-input preparation and is not presented as a parameter of the published baseline-correction method.
## Reference
Zhou, B., Zhang, C., Wang, X., Wang, Y., Yin, Y., & Gong, M. (2026). A VMD-based baseline correction method of strong ground motion records and permanent displacement evaluation. *Soil Dynamics and Earthquake Engineering, 203*, 110089. [https://doi.org/10.1016/j.soildyn.2026.110089](https://doi.org/10.1016/j.soildyn.2026.110089).

Please cite this article when using the permanent-displacement processing method, together with the original strong-motion data provider and the FRONT/PI-FRONT paper where applicable.
