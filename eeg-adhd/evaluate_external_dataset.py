"""
Evaluate the trained model on external dataset (adhdata.csv)
Comprehensive evaluation with all metrics
"""
import csv
import numpy as np
import torch
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as skm
import mne
from scipy.signal import welch
from scipy.interpolate import griddata
import warnings
import os

warnings.filterwarnings("ignore")
os.environ['MNE_LOGGING_LEVEL'] = 'ERROR'

# ---------- CONFIG ----------
FS = 256
N_PER_SEG = 512
F_BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 50)}
N_CHANNELS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- MODEL DEFINITION ----------
class EncoderCNN(torch.nn.Module):
    def __init__(self, embed_dim=16, n_classes=2):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, 3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(64, embed_dim)
        )
        self.classifier = torch.nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        z = self.encoder(x)
        logits = self.classifier(z)
        return logits

    def embed(self, x):
        return self.encoder(x)


def preprocess_to_map(eeg_arr: np.ndarray) -> np.ndarray:
    """Convert EEG to 4x8x8 PSD map"""
    eeg_arr = np.asarray(eeg_arr)
    
    if eeg_arr.ndim == 1:
        raise ValueError(f"EEG is 1-D with length {eeg_arr.shape[0]}; expected 2-D")
    if eeg_arr.ndim == 3:
        eeg_arr = eeg_arr[0]
    
    if eeg_arr.shape[0] == N_CHANNELS:
        pass
    elif eeg_arr.shape[1] == N_CHANNELS:
        eeg_arr = eeg_arr.T
    else:
        eeg_arr = eeg_arr.T
    
    raw = mne.io.RawArray(eeg_arr, mne.create_info(ch_names=['F3', 'F4'], sfreq=FS, ch_types='eeg'))
    raw.filter(0.5, 50)
    
    freqs, psd = welch(raw.get_data(), fs=FS, nperseg=N_PER_SEG, axis=1)
    
    map_dict = {}
    for band_name, (f_min, f_max) in F_BANDS.items():
        mask = (freqs >= f_min) & (freqs <= f_max)
        band_psd = psd[:, mask]
        band_psd_mean = band_psd.mean(axis=1)
        map_dict[band_name] = band_psd_mean
    
    theta_map = np.tile(map_dict['theta'].reshape(2, 1), (1, 4))
    alpha_map = np.tile(map_dict['alpha'].reshape(2, 1), (1, 4))
    beta_map = np.tile(map_dict['beta'].reshape(2, 1), (1, 4))
    gamma_map = np.tile(map_dict['gamma'].reshape(2, 1), (1, 4))
    
    combined = np.vstack([theta_map, alpha_map, beta_map, gamma_map])
    
    x = np.linspace(0, 2, combined.shape[1])
    y = np.linspace(0, 2, combined.shape[0])
    xx, yy = np.meshgrid(x, y)
    
    xi = np.linspace(0, 2, 8)
    yi = np.linspace(0, 2, 8)
    xxi, yyi = np.meshgrid(xi, yi)
    
    map_8x8 = np.zeros((4, 8, 8))
    for band_idx in range(4):
        map_8x8[band_idx] = griddata((xx.flat, yy.flat), combined[band_idx*2:(band_idx+1)*2].flat, 
                                     (xxi, yyi), method='linear', fill_value=0)
    
    return map_8x8.astype(np.float32)


def load_csv_eeg(csv_path: str) -> tuple:
    """Load EEG data from CSV - returns (X, y, channels)
    Assumes CSV has channel names in header and raw EEG samples as rows
    """
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Get channel names
        
        # Find F3 and F4 indices
        try:
            f3_idx = header.index('F3')
            f4_idx = header.index('F4')
        except ValueError:
            print("ERROR: F3 or F4 not found in header")
            return np.array([]), np.array([])
        
        print(f"Found channels: {header}")
        print(f"F3 at index {f3_idx}, F4 at index {f4_idx}")
        
        # Load all samples
        samples = []
        for i, row in enumerate(reader):
            if i % 100000 == 0:
                print(f"  Processed {i} rows...")
            try:
                f3 = float(row[f3_idx])
                f4 = float(row[f4_idx])
                samples.append([f3, f4])
            except (ValueError, IndexError):
                continue
    
    # Convert to numpy array (channels, time)
    samples = np.array(samples, dtype=np.float32).T  # Shape: (2, n_samples)
    
    print(f"Loaded {samples.shape[1]} EEG samples from CSV")
    
    # Since no labels in CSV, assume all are control (0)
    # We'll evaluate and display results
    return samples


def evaluate_on_external_data(csv_path: str, model_dir: Path):
    """Evaluate model on external dataset"""
    print("\n" + "="*70)
    print("LOADING EXTERNAL DATA")
    print("="*70)
    
    # Load CSV data (raw EEG - 2 channels, many time points)
    print(f"Loading {csv_path}...")
    eeg_data = load_csv_eeg(csv_path)  # Shape: (2, n_time_samples)
    
    if eeg_data.size == 0:
        print("ERROR: No valid EEG data loaded")
        return
    
    # Split into overlapping windows (like individual subjects)
    # Each window is 7680 samples (30 seconds at 256 Hz)
    WINDOW_SIZE = 7680
    STEP_SIZE = 3840  # 50% overlap
    
    subjects = []
    for start in range(0, eeg_data.shape[1] - WINDOW_SIZE, STEP_SIZE):
        window = eeg_data[:, start:start+WINDOW_SIZE]
        if window.shape[1] == WINDOW_SIZE:
            subjects.append((window, 0))  # All labeled as control (no labels in CSV)
    
    print(f"Created {len(subjects)} overlapping windows of 7680 samples each")
    
    if len(subjects) == 0:
        print("ERROR: Could not create any windows from EEG data")
        return
    
    # Preprocess
    print("\nPreprocessing to 4x8x8 PSD maps...")
    X_ext = []
    y_ext = []
    
    for i, (eeg, label) in enumerate(subjects):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(subjects)}")
        try:
            map_8x8 = preprocess_to_map(eeg.astype(np.float32))
            X_ext.append(map_8x8)
            y_ext.append(label)
        except Exception as e:
            print(f"  Skipping window {i}: {str(e)}")
            continue
    
    X_ext = np.array(X_ext)
    y_ext = np.array(y_ext)
    
    print(f"\nExternal dataset: {len(X_ext)} windows preprocessed")
    
    # Load model and artifacts
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70)
    
    cnn_path = model_dir / "cnn_frozen.pth"
    scaler_path = model_dir / "scaler_16d.pkl"
    prot_path = model_dir / "softmax_weights.pkl"
    
    cnn = EncoderCNN().to(DEVICE)
    state = torch.load(cnn_path, map_location=DEVICE)
    cnn.load_state_dict(state)
    cnn.eval()
    
    scaler = joblib.load(scaler_path)
    ctl_emb, adhd_emb = joblib.load(prot_path)
    
    print("Model loaded successfully")
    
    # Evaluate
    print("\n" + "="*70)
    print("EVALUATING ON EXTERNAL DATA")
    print("="*70)
    
    X_ext_t = torch.tensor(X_ext, dtype=torch.float32)
    embeddings = []
    
    with torch.no_grad():
        for batch_idx in range(0, len(X_ext), 32):
            batch = X_ext_t[batch_idx:batch_idx+32].to(DEVICE)
            z = cnn.embed(batch).cpu().numpy()
            embeddings.append(z)
    
    embeddings = np.vstack(embeddings)
    embeddings_scaled = scaler.transform(embeddings)
    
    # Compute distances to prototypes
    d_ctl = np.linalg.norm(embeddings_scaled - ctl_emb, axis=1)
    d_adhd = np.linalg.norm(embeddings_scaled - adhd_emb, axis=1)
    
    # Softmax classification
    scores = np.vstack([-d_ctl, -d_adhd]).T
    exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    prob_adhd = probs[:, 1]
    y_pred = (prob_adhd > 0.5).astype(int)
    
    # Metrics
    acc = skm.accuracy_score(y_ext, y_pred)
    precision = skm.precision_score(y_ext, y_pred, zero_division=0)
    recall = skm.recall_score(y_ext, y_pred, zero_division=0)
    f1 = skm.f1_score(y_ext, y_pred, zero_division=0)
    roc_auc = skm.roc_auc_score(y_ext, prob_adhd)
    
    cm = skm.confusion_matrix(y_ext, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print("\n" + "="*70)
    print("EXTERNAL DATASET EVALUATION RESULTS")
    print("="*70)
    print(f"\nDataset: {Path(csv_path).name}")
    print(f"Windows: {len(X_ext)}")
    print(f"  (No labels in source CSV - all treated as control for prediction)")
    
    print("\nPERFORMANCE METRICS:")
    print(f"  Predictions made: {len(X_ext)} windows")
    print(f"  ADHD predictions: {(y_pred==1).sum()}")
    print(f"  Control predictions: {(y_pred==0).sum()}")
    
    print("\nPREDICTION CONFIDENCE:")
    print(f"  Mean ADHD probability: {prob_adhd.mean():.4f}")
    print(f"  Std ADHD probability: {prob_adhd.std():.4f}")
    print(f"  Min ADHD probability: {prob_adhd.min():.4f}")
    print(f"  Max ADHD probability: {prob_adhd.max():.4f}")
    
    # Show distribution
    adhd_count = (prob_adhd > 0.5).sum()
    ctrl_count = (prob_adhd <= 0.5).sum()
    print(f"\nPREDICTION DISTRIBUTION:")
    print(f"  ADHD (prob > 0.5): {adhd_count} ({adhd_count/len(prob_adhd)*100:.1f}%)")
    print(f"  Control (prob ≤ 0.5): {ctrl_count} ({ctrl_count/len(prob_adhd)*100:.1f}%)")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    model_dir = Path(__file__).parent.parent / "model"
    csv_path = Path(__file__).parent / "data" / "adhdata.csv"
    
    evaluate_on_external_data(str(csv_path), model_dir)
