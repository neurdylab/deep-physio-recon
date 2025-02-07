## Physiological Signal Reconstruction using Deep Learning

### Description

Functional magnetic resonance imaging (fMRI) is a powerful technique for studying human brain activity and large-scale neural circuits. However, fMRI signals can be strongly modulated by slow changes in respiration volume (RV) and heart rate (HR). Monitoring cardiac and respiratory signals during fMRI enables modeling and/or reducing such effects; yet, physiological measurements are often unavailable in practice, and are missing from a large number of fMRI datasets. Here, we propose learning approaches for inferring RV and HR signals directly from fMRI time-series dynamics. 

![Method overview](signals.jpg)

### Code Organization

This repository contains implementations for several published works on physiological signal reconstruction from fMRI data. Please note:

- **Active Development**: The code for our latest work (JOURNAL-NAME-2024) is actively maintained and can be found in the `JOURNAL-NAME-2024` directory.
- **Legacy Code**: Implementation for previous publications is provided for reference but is no longer actively maintained.

### Publications

1. ["Tracing peripheral physiology in low frequency fMRI dynamics"](https://osf.io/preprints/osf/fj4gq_v1) - Preprint (2024)

2. ["From Brain to Body: Learning Low-Frequency Respiration and Cardiac Signals from fMRI Dynamics"](https://link.springer.com/chapter/10.1007/978-3-030-87234-2_52) - MICCAI (2021)

3. ["A Deep Pattern Recognition Approach for Inferring Respiratory Volume Fluctuations from fMRI Data"](https://link.springer.com/chapter/10.1007/978-3-030-59728-3_42) - MICCAI (2020)

4. ["Reconstruction of respiratory variation signals from fMRI data"](https://doi.org/10.1016/j.neuroimage.2020.117459) - Neuroimage (2020)

### Getting Started

For the latest implementation and usage instructions, please refer to the documentation in the `JOURNAL-NAME-2024` directory.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Contact

roza.g.bayrak@vanderbilt.edu
