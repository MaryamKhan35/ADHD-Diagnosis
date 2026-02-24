"""
Extract a single subject from FC.mat and save to a separate file
"""
import scipy.io as sio
import numpy as np
from pathlib import Path

# Load FC.mat
fc_path = Path(__file__).parent / "data" / "FC.mat"
print(f"Loading FC.mat from: {fc_path}")

mat = sio.loadmat(str(fc_path))

# Find the EEG data
eeg_data = None
for key, value in mat.items():
    if key.startswith('__'):
        continue
    print(f"Key: {key}, Type: {type(value)}, Shape: {getattr(value, 'shape', 'N/A')}")
    
    if isinstance(value, np.ndarray) and value.dtype == object:
        print(f"  Object array found")
        for item in value.flat:
            if isinstance(item, np.ndarray):
                print(f"    Item shape: {item.shape}")
                if item.ndim == 3:
                    print(f"    3D array detected - extracting first subject")
                    eeg_data = item[0]  # Get first subject
                    break
        if eeg_data is not None:
            break
    elif isinstance(value, np.ndarray) and value.ndim == 3:
        print(f"  3D array detected - extracting first subject")
        eeg_data = value[0]  # Get first subject
        break

if eeg_data is not None:
    print(f"\nExtracted single subject with shape: {eeg_data.shape}")
    
    # Save to new file
    output_path = Path(__file__).parent / "data" / "FC_subject_1.mat"
    sio.savemat(str(output_path), {'eeg': eeg_data})
    print(f"Saved to: {output_path}")
else:
    print("No EEG data found!")
