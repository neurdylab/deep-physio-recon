## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/neurdylab/deep-physio-recon.git
   cd deep-physio-recon/JOURNAL-NAME-2024
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv .env
   source .env/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

1. **Prepare Your Data**
   - Follow the preprocessing steps detailed in [preprocessing/README.md](../preprocessing/README.md). This will ensure your data is properly processed.

2. **Download Pre-trained Models**
   ```python
   # Install huggingface_hub
   pip install huggingface_hub

   # In Python, download the weights:
   from huggingface_hub import hf_hub_download
   file = hf_hub_download(
      repo_id="rgbayrak/deep-physio-recon",
      filename="Bi-LSTM_schaefertractsegtianaan_lr_0.001_l1_0.5/saved_model_split_train_fold_0",
      local_dir="./weights",
      local_dir_use_symlinks=False 
   )
   print(f"Downloaded model: {file}")
   ```

   The models are also available directly on the [HuggingFace Model Hub](https://huggingface.co/rgbayrak/deep-physio-recon).

3. **Run Demo**
   ```bash
   python run_demo.py
   ```
   This will run the model on example data provided in the repository, demonstrating the expected inputs and outputs. The following files will be generated in the output directory:

   - **Reconstructed Signals** (`<subject_id>_pred.mat`):
      - `RV`: Reconstructed respiratory volume signal
      - `HR`: Reconstructed heart rate signal

   - **Visualizations** (`<subject_id>_QA.png`):
      - Quality assessment plot of reconstructed RV and HR signals
   
   - **Quality Metrics** (`subject_metrics.json`):
      - Signal quality metrics

4. **Run Inference on Your Data**
   
   First, organize your data in the following structure:
   ```
   data/
   └── fmri/
       ├── subject1.mat
       ├── subject2.mat
       └── subject3.mat
   ```

   Then run:
   ```bash
   python run_inference.py --input_dir data --output_dir results
   ```
   For each subject in the `data` directory, the following files will be generated in the output directory:

   - **Reconstructed Signals** (`<subject_id>_pred.mat`):

      - RV: Reconstructed respiratory volume signal
      - HR: Reconstructed heart rate signal

   - **Visualizations** (`<subject_id>_QA.png`):
      - Visualization of reconstructed RV and HR signals

## Citation
If you find this work useful, please cite:

```bibtex
@article{bayraktracing,
  title={Tracing peripheral physiology in low frequency fMRI dynamics},
  author={Bayrak, Roza Gunes and Hansen, Colin and Salas, Jorge and Ahmed, Nafis and Lyu, Ilwoo and Mather, Mara and Huo, Yuankai and Chang, Catie},
  publisher={OSF}
}
```