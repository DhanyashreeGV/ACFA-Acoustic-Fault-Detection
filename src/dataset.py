import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from src.utils import resolve_filepath, standardize_label, CLASS_TO_IDX
from src.preprocessing import process_file

class AcousticFaultDataset(Dataset):
    """
    PyTorch Dataset for ACFA Acoustic Fault Detection.
    Loads audio, applies preprocessing pipeline, and yields Mel-Spectrogram tensors.
    """
    def __init__(
        self,
        csv_path: str,
        base_dir: str = None,
        transform = None,
        segment_level: bool = False
    ):
        self.csv_path = csv_path
        self.base_dir = base_dir
        self.transform = transform
        self.segment_level = segment_level
        
        self.df = pd.read_csv(csv_path)
        self.samples = []
        
        for _, row in self.df.iterrows():
            raw_path = str(row['filepath'])
            local_path = resolve_filepath(raw_path, base_dir=self.base_dir)
            std_label = standardize_label(str(row['label']))
            label_idx = CLASS_TO_IDX[std_label]
            self.samples.append((local_path, label_idx, std_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        filepath, label_idx, std_label = self.samples[idx]
        
        try:
            # Apply complete preprocessing pipeline
            spectrograms = process_file(filepath)
            # Take middle or primary segment spectrogram (shape: 64, 61)
            spec = spectrograms[len(spectrograms) // 2]
        except Exception as e:
            print(f"Warning: Failed to process audio at '{filepath}': {e}. Using zero-spectrogram fallback.")
            spec = np.zeros((64, 61), dtype=np.float32)
        
        # Convert to float32 Tensor with channel dimension (1, n_mels, time_frames)
        tensor_spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
        
        if self.transform:
            tensor_spec = self.transform(tensor_spec)
            
        return tensor_spec, torch.tensor(label_idx, dtype=torch.long)


def create_dataloader(
    csv_path: str,
    base_dir: str = None,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """Helper function to instantiate DataLoader."""
    dataset = AcousticFaultDataset(csv_path=csv_path, base_dir=base_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
