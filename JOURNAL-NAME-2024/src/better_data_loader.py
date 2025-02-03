from pathlib import Path
import numpy as np
import scipy.io as sio
from typing import Dict, Union, Optional
import torch
from torch.utils.data import Dataset, DataLoader


## NOT TESTED
class Dataset(Dataset):
    """Dataset class for physiological and fMRI data."""
    def __init__(self, 
                 data_path: Union[str, Path],
                 mode: str = 'train',
                 transform=None):
        """
        Args:
            data_path: Path to data directory or single file
            mode: 'train' or 'inference'
            transform: Optional transform to be applied
        """
        self.data_path = Path(data_path)
        self.mode = mode
        self.transform = transform
        
        # Load data
        self.data = self._load_data()
        
    def _load_data(self) -> Dict:
        """Load and validate data."""
        try:
            data = sio.loadmat(self.data_path)
            
            # Validate required fields
            if 'timeseries' not in data:
                raise ValueError("Missing required field: timeseries")
                
            if self.mode == 'train':
                if 'RV' not in data or 'HR' not in data:
                    raise ValueError("Training mode requires RV and HR signals")
            
            return data
            
        except Exception as e:
            raise IOError(f"Error loading {self.data_path}: {str(e)}")
    
    def __len__(self) -> int:
        return 1  # Single subject
    
    def __getitem__(self, idx: int) -> Dict:
        """Get the data with optional transform."""
        sample = {
            'fmri': torch.FloatTensor(self.data['timeseries'])
        }
        
        if self.mode == 'train':
            sample.update({
                'RV': torch.FloatTensor(self.data['RV']),
                'HR': torch.FloatTensor(self.data['HR'])
            })
            
        if self.transform:
            sample = self.transform(sample)
            
        return sample

def get_dataloader(data_path: Union[str, Path],
                  mode: str = 'train',
                  transform=None) -> DataLoader:
    """Create DataLoader for training or inference.
    
    Args:
        data_path: Path to data file
        mode: 'train' or 'inference'
        transform: Optional transform to be applied
        
    Returns:
        DataLoader instance
    """
    dataset = Dataset(
        data_path=data_path,
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
    """Normalize fMRI and physiological signals."""
    def __call__(self, sample: Dict) -> Dict:
        sample['fmri'] = (sample['fmri'] - sample['fmri'].mean(0)) / sample['fmri'].std(0)
        if 'RV' in sample:
            sample['RV'] = (sample['RV'] - sample['RV'].mean()) / sample['RV'].std()
        if 'HR' in sample:
            sample['HR'] = (sample['HR'] - sample['HR'].mean()) / sample['HR'].std()
        return sample