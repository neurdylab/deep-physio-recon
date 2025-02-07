from pathlib import Path
import numpy as np
import scipy.io as sio
from typing import Dict, Union, Optional, List
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.stats import zscore


class Dataset(Dataset):
    """Dataset class for physiological and fMRI data."""
    def __init__(self, 
                 data_path: Union[str, Path],
                 roi_list: List[str] = ['schaefer', 'tractseg', 'tian', 'aan'],
                 mode: str = 'train',
                 transform=None):
        """
        Args:
            data_path: Path to base data directory (e.g., example_data)
            roi_list: List of atlas names to load and concatenate
            mode: 'train' or 'inference'
            transform: Optional transform to be applied
        """
        self.data_path = Path(data_path)
        self.roi_list = roi_list
        self.mode = mode
        self.transform = transform
        
        # Load data
        self.data = self._load_data()
        
        # Store input size (number of ROIs) from first subject
        if not self.data:
            raise ValueError("No valid data loaded")
        self.input_size = self.data[0]['timeseries'].shape[1]
        
    def _load_data(self) -> Dict:
        """Load and concatenate data from multiple atlases."""
        try:
            # Get list of subject files from first atlas to determine available subjects
            first_atlas_dir = self.data_path / 'fMRI' / self.roi_list[0]
            if not first_atlas_dir.exists():
                raise FileNotFoundError(f"Atlas directory not found: {first_atlas_dir}")
            
            # Get all subject files
            subject_files = sorted(list(first_atlas_dir.glob('*.mat')))
            if not subject_files:
                raise FileNotFoundError(f"No .mat files found in {first_atlas_dir}")
            
            # Process all subjects
            all_subjects_data = []
            for subject_file in subject_files:
                # For fMRI files, keep the full name (with roi_ prefix if it exists)
                fmri_id = subject_file.stem
                # For HR/RV files, remove roi_ prefix if it exists
                subject_id = fmri_id.replace('rois_', '')
                
                all_timeseries = []  # Reset for each subject
                
                # Load and concatenate each atlas's timeseries for this subject
                for atlas in self.roi_list:
                    atlas_file = self.data_path / 'fMRI' / atlas / f"{fmri_id}.mat"
                    if not atlas_file.exists():
                        print(f"Warning: Missing atlas file: {atlas_file}, skipping subject {subject_id}")
                        break
                    
                    atlas_data = sio.loadmat(atlas_file)
                    if 'roi_dat' not in atlas_data:
                        print(f"Warning: Missing roi_dat in {atlas_file}, skipping subject {subject_id}")
                        break
                    
                    all_timeseries.append(atlas_data['roi_dat'])
                
                # Only add subject if all atlas data was loaded successfully
                if len(all_timeseries) == len(self.roi_list):
                    combined_timeseries = np.concatenate(all_timeseries, axis=1)
                    all_subjects_data.append({
                        'timeseries': combined_timeseries,
                        'subject_id': subject_id,  # Clean ID for HR/RV files
                        'fmri_id': fmri_id  # Original ID for fMRI files
                    })
            
            if not all_subjects_data:
                raise ValueError("No valid subjects found with complete data")
                
            return all_subjects_data
            
        except Exception as e:
            raise IOError(f"Error loading data: {str(e)}")
    
    def __len__(self) -> int:
        return len(self.data)  # Number of subjects
    
    def __getitem__(self, idx: int) -> Dict:
        """Get the data with optional transform."""
        # Get data for specific subject
        subject_data = self.data[idx]
        subject_id = subject_data['subject_id']
        
        # Transpose from (time x channels) to (channels x time)
        timeseries = subject_data['timeseries'].transpose((1, 0))
        
        sample = {
            'fmri': torch.FloatTensor(timeseries),
            'subject_id': subject_id
        }
        
        if self.mode == 'train':
            # Load HR and RV data with the correct file naming pattern and paths
            hr_file = self.data_path / "HR" / f"{subject_id}_rfMRI_REST1_LR_hr_filt_ds.mat"
            rv_file = self.data_path / "RV" / f"{subject_id}_rfMRI_REST1_LR_rv_filt_ds.mat"
            
            if not hr_file.exists() or not rv_file.exists():
                raise FileNotFoundError(f"Missing physio files for subject {subject_id}")
            
            hr_data = sio.loadmat(hr_file)
            rv_data = sio.loadmat(rv_file)
            
            if 'hr_filt_ds' not in hr_data or 'rv_filt_ds' not in rv_data:
                raise ValueError(f"Missing required signals in physio files for subject {subject_id}")
            
            sample.update({
                'HR': torch.FloatTensor(hr_data['hr_filt_ds']),
                'RV': torch.FloatTensor(rv_data['rv_filt_ds'])
            })
            
        if self.transform:
            sample = self.transform(sample)
        return sample

def get_dataloader(data_path: Union[str, Path],
                  roi_list: List[str] = ['schaefer', 'tractseg', 'tian', 'aan'],
                  mode: str = 'train',
                  transform=None) -> DataLoader:
    """Create DataLoader for training or inference.
    
    Args:
        data_path: Path to base data directory
        roi_list: List of atlas names to load
        mode: 'train' or 'inference'
        transform: Optional transform to be applied
        
    Returns:
        DataLoader instance
    """
    dataset = Dataset(
        data_path=data_path,
        roi_list=roi_list,
        mode=mode,
        transform=transform
    )
    
    return DataLoader(
        dataset,
        batch_size=1,  # Single subject
        shuffle=False
    )

# Optional: Data transforms
class Normalize:
    """Normalize fMRI and physiological signals using z-score along time axis."""
    def __call__(self, sample: Dict) -> Dict:
        # fMRI data: (ROIs x time)
        sample['fmri'] = torch.FloatTensor(zscore(sample['fmri'].numpy(), axis=1))
        
        # Physiological signals
        if 'RV' in sample:
            sample['RV'] = torch.FloatTensor(zscore(sample['RV'].numpy(), axis=1))
        if 'HR' in sample:
            sample['HR'] = torch.FloatTensor(zscore(sample['HR'].numpy(), axis=1))
        
        return sample