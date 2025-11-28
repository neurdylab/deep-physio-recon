## Physiological Signal Reconstruction using Deep Learning

### Description

Many studies of the human brain using functional magnetic resonance imaging (fMRI) lack physiological measurements, which substantially impacts the interpretation and richness of fMRI studies. Natural fluctuations in autonomic physiology, such as breathing and heart rate, provide windows into critical functions, including cognition, emotion, and health, and can heavily influence fMRI signals. Here, we developed DeepPhysioRecon, a Long-Short-Term-Memory (LSTM)-based network that decodes continuous variations in respiration amplitude and heart rate directly from whole-brain fMRI dynamics. Through systematic evaluations, we investigate the generalizability of this approach across datasets and experimental conditions. We also demonstrate the importance of including these measures in fMRI analyses. This work highlights the importance of studying brain-body interactions, proposes a tool that may enhance the efficacy of fMRI as a biomarker, and provides widely applicable open-source software.

![Method overview](signals.jpg)

### Code Organization

This repository contains implementations for several published works on physiological signal reconstruction from fMRI data. Please note:

- **Active Development**: The code for our latest work is actively maintained and can be found in the `IMAGINGNEURO2025` directory.
- **Legacy Code**: Implementation for previous publications is provided for reference but is no longer actively maintained.

### Publications

1. ["Deep Physio Recon: Tracing peripheral physiology in low frequency fMRI dynamics"](https://direct.mit.edu/imag/article/doi/10.1162/IMAG.a.163/133034) - Imaging Neuroscience (2025)

2. ["From Brain to Body: Learning Low-Frequency Respiration and Cardiac Signals from fMRI Dynamics"](https://link.springer.com/chapter/10.1007/978-3-030-87234-2_52) - MICCAI (2021)

3. ["A Deep Pattern Recognition Approach for Inferring Respiratory Volume Fluctuations from fMRI Data"](https://link.springer.com/chapter/10.1007/978-3-030-59728-3_42) - MICCAI (2020)

4. ["Reconstruction of respiratory variation signals from fMRI data"](https://doi.org/10.1016/j.neuroimage.2020.117459) - Neuroimage (2020)


### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Contact

roza.g.bayrak@vanderbilt.edu
