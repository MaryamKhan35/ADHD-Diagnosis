"""
Evaluate the current trained model on all subjects from FC, MC, FADHD, MADHD
"""
import scipy.io as sio
import numpy as np
import torch
import joblib
from pathlib import Path
import mne
from scipy.signal import welch
from scipy.interpolate import griddata
import sklearn.metrics as skm
import warnings
import os

warnings.filterwarnings("ignore")
os.environ['MNE_LOGGING_LEVEL'] = 'ERROR'

# Config (must match training)
FS = 256
N_PER_SEG = 512
F_BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 50)}
N_CHANNELS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Encoder model
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


def preprocess_to_map(eeg_arr):
    """eeg_arr: (channels, time) -> (4, 8, 8) PSD map"""
    info = mne.create_info(
        ch_names=[f"Ch{i}" for i in range(N_CHANNELS)], 
        sfreq=FS, 
        ch_types=["eeg"] * N_CHANNELS
    )
    raw = mne.io.RawArray(eeg_arr, info)
    raw.filter(l_freq=0.5, h_freq=50., fir_design="firwin", verbose=False)
    raw.notch_filter(50., verbose=False)

    try:
        raw.set_montage("standard_1005", verbose=False)
        montage = mne.channels.make_standard_montage("standard_1005")
        ch_pos_map = montage.get_positions()["ch_pos"]
        pos = []
        for ch in raw.ch_names:
            p = ch_pos_map.get(ch)
            if p is None:
                raise KeyError
            pos.append(p)
        pos = np.array(pos)
    except Exception:
        angles = np.linspace(0, 2 * np.pi, N_CHANNELS, endpoint=False)
        pos = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(N_CHANNELS)])

    clean = raw.get_data()
    freqs, psd = welch(clean, fs=FS, nperseg=N_PER_SEG, axis=-1)
    band_power = []
    for low, high in F_BANDS.values():
        idx = np.logical_and(freqs >= low, freqs <= high)
        band_power.append(psd[:, idx].mean(axis=-1))
    bp = np.array(band_power)

    x_grid = np.linspace(pos[:, 0].min(), pos[:, 0].max(), 8)
    y_grid = np.linspace(pos[:, 1].min(), pos[:, 1].max(), 8)
    xx, yy = np.meshgrid(x_grid, y_grid)
    map_8x8 = np.zeros((4, 8, 8))
    for b in range(4):
        try:
            grid_lin = griddata(pos[:, :2], bp[b], (xx, yy), method="linear")
            if np.isnan(grid_lin).any():
                grid_nearest = griddata(pos[:, :2], bp[b], (xx, yy), method="nearest")
                grid_lin = np.where(np.isnan(grid_lin), grid_nearest, grid_lin)
        except Exception:
            grid_lin = griddata(pos[:, :2], bp[b], (xx, yy), method="nearest")
        map_8x8[b] = grid_lin
    return map_8x8


def load_all_subjects(mat_paths):
    """Load all subjects from mat files"""
    all_subj = []
    labels = [0, 0, 1, 1]  # FC=0, MC=0, FADHD=1, MADHD=1
    names = ["FC (Control Female)", "MC (Control Male)", "FADHD (ADHD Female)", "MADHD (ADHD Male)"]
    
    for path, label, name in zip(mat_paths, labels, names):
        print("Loading {}...".format(name))
        mat = sio.loadmat(path)
        subject_count = 0
        
        for k, v in mat.items():
            if k.startswith('__'):
                continue
            if not isinstance(v, np.ndarray):
                continue

            if v.dtype == np.object_:
                for item in v.flat:
                    if not isinstance(item, np.ndarray):
                        continue
                    if item.ndim == 3:
                        n_subj = item.shape[0]
                        for s in range(n_subj):
                            subj = item[s]
                            if subj.ndim != 2:
                                continue
                            if subj.shape[1] == N_CHANNELS:
                                eeg = subj.T
                            elif subj.shape[0] == N_CHANNELS:
                                eeg = subj
                            else:
                                eeg = subj.T
                            if not np.isfinite(eeg).all():
                                continue
                            # Skip corrupted subject 7 in FADHD (0-indexed as 6)
                            if label == 1 and name.startswith("FADHD") and s == 6:
                                print("  Skipping corrupted subject 7")
                                continue
                            all_subj.append((eeg.astype(np.float32), label))
                            subject_count += 1
                    elif item.ndim == 2:
                        if item.shape[1] == N_CHANNELS:
                            eeg = item.T
                        elif item.shape[0] == N_CHANNELS:
                            eeg = item
                        else:
                            eeg = item.T
                        if not np.isfinite(eeg).all():
                            continue
                        all_subj.append((eeg.astype(np.float32), label))
                        subject_count += 1
        
        print("  Loaded {} subjects".format(subject_count))
    
    return all_subj


# --- MAIN ---
print("\n" + "="*70)
print("EVALUATING CURRENT MODEL")
print("="*70)

# Load data
script_dir = Path(__file__).parent
mat_files = [
    script_dir / "data" / "FC.mat",
    script_dir / "data" / "MC.mat",
    script_dir / "data" / "FADHD.mat",
    script_dir / "data" / "MADHD.mat"
]
mat_files = [str(p) for p in mat_files]

print("\nLoading subjects...")
all_subj = load_all_subjects(mat_files)
print("\nTotal subjects loaded: {}".format(len(all_subj)))

# Count by label
n_control = sum(1 for _, label in all_subj if label == 0)
n_adhd = sum(1 for _, label in all_subj if label == 1)
print("  Control: {}".format(n_control))
print("  ADHD:    {}".format(n_adhd))

# Preprocess to maps
print("\nPreprocessing to PSD maps...")
X_full = []
y_full = []
for i, (eeg, label) in enumerate(all_subj):
    if (i + 1) % 20 == 0:
        print("  Processed {}/{}".format(i + 1, len(all_subj)))
    try:
        map_8x8 = preprocess_to_map(eeg)
        X_full.append(map_8x8)
        y_full.append(label)
    except Exception as e:
        print("  ERROR processing sample {}: {}".format(i, e))
        continue

X_full = np.array(X_full)
y_full = np.array(y_full)

print("Final data shape: {}".format(X_full.shape))

# Load model files
model_dir = script_dir.parent / "model"
print("\nLoading model from {}...".format(model_dir))

cnn = EncoderCNN().to(DEVICE)
cnn.load_state_dict(torch.load(str(model_dir / "cnn_frozen.pth"), map_location=DEVICE))
cnn.eval()

prot = joblib.load(str(model_dir / "softmax_weights.pkl"))
scaler = joblib.load(str(model_dir / "scaler_16d.pkl"))

print("Model loaded successfully")

# Get embeddings
print("\nExtracting embeddings...")
X_full_t = torch.tensor(X_full, dtype=torch.float32)
loader = torch.utils.data.DataLoader(X_full_t, batch_size=32, shuffle=False)

embeddings = []
with torch.no_grad():
    for xb in loader:
        z = cnn.embed(xb.to(DEVICE)).cpu().numpy()
        embeddings.append(z)
embeddings = np.vstack(embeddings)

print("Embeddings shape: {}".format(embeddings.shape))

# Scale and predict
embeddings_scaled = scaler.transform(embeddings)
ctl_emb, adhd_emb = prot

d_ctl = np.linalg.norm(embeddings_scaled - ctl_emb, axis=1)
d_adhd = np.linalg.norm(embeddings_scaled - adhd_emb, axis=1)

scores = np.vstack([-d_ctl, -d_adhd]).T
exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
probs = exp / exp.sum(axis=1, keepdims=True)
prob_adhd = probs[:, 1]
y_pred = (prob_adhd > 0.5).astype(int)

# Calculate metrics
print("\n" + "="*70)
print("EVALUATION METRICS")
print("="*70)

acc = skm.accuracy_score(y_full, y_pred)
precision = skm.precision_score(y_full, y_pred, zero_division=0)
recall = skm.recall_score(y_full, y_pred, zero_division=0)
f1 = skm.f1_score(y_full, y_pred, zero_division=0)
sensitivity = recall
specificity = skm.recall_score(y_full, 1-y_pred, pos_label=0, zero_division=0)

try:
    roc_auc = skm.roc_auc_score(y_full, prob_adhd)
except:
    roc_auc = 0.0

cm = skm.confusion_matrix(y_full, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\nCLASSIFICATION METRICS:")
print("  Accuracy:    {:.4f} ({:.2f}%)".format(acc, acc*100))
print("  Precision:   {:.4f}".format(precision))
print("  Recall:      {:.4f}".format(recall))
print("  Sensitivity: {:.4f}".format(sensitivity))
print("  Specificity: {:.4f}".format(specificity))
print("  F1-Score:    {:.4f}".format(f1))
print("  ROC-AUC:     {:.4f}".format(roc_auc))

print("\nCONFUSION MATRIX:")
print("                  Predicted")
print("              Control  ADHD")
print("  Actual Control    {}    {}".format(tn, fp))
print("  Actual ADHD       {}    {}".format(fn, tp))

print("\nPREDICTION DISTRIBUTION:")
print("  Predicted Control: {}".format(sum(y_pred==0)))
print("  Predicted ADHD:    {}".format(sum(y_pred==1)))
print("  Actual Control:    {}".format(sum(y_full==0)))
print("  Actual ADHD:       {}".format(sum(y_full==1)))

print("\nPROBABILITY STATISTICS (ADHD class):")
print("  Min:  {:.4f}".format(prob_adhd.min()))
print("  Max:  {:.4f}".format(prob_adhd.max()))
print("  Mean: {:.4f}".format(prob_adhd.mean()))
print("  Std:  {:.4f}".format(prob_adhd.std()))

print("\n" + "="*70)
print("SUMMARY: This model accuracy is {:.2f}%".format(acc*100))
print("="*70)
