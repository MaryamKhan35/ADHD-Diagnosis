"""
train_weights.py
Train CNN on your 528 EEGs → save real prototypes + soft-max weights
No external data needed uses your 4 .mat cell arrays
"""
import scipy.io as sio
import numpy as np
import torch
import joblib
from sklearn.preprocessing import StandardScaler
import mne
from scipy.signal import welch
from scipy.interpolate import griddata
import sklearn.metrics as skm
from sklearn.model_selection import KFold
import tqdm
from pathlib import Path

import warnings, os, logging
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
os.environ['MNE_LOGGING_LEVEL'] = 'ERROR'

# ---------- CONFIG ----------
FS       = 256
N_PER_SEG = 512
F_BANDS  = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 50)}
N_CHANNELS = 2                  # Cz, F4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- 1. LOAD ALL SUBJECTS ----------
def load_all_subjects(mat_paths):
    """Returns list of (eeg, label)  eeg=(channels, time)  label=0/1"""
    all_subj = []
    for path, label in zip(mat_paths, [0, 0, 1, 1]):  # FC=0, MC=0, FADHD=1, MADHD=1
        mat = sio.loadmat(path)
        # Find candidate arrays (either cell arrays or direct ndarrays)
        for k, v in mat.items():
            if k.startswith('__'):
                continue
            if not isinstance(v, np.ndarray):
                continue

            # cell array: iterate elements
            if v.dtype == np.object_:
                for item in v.flat:
                    if not isinstance(item, np.ndarray):
                        continue
                    # item may be shape (n_subjects, n_samples, n_channels)
                    if item.ndim == 3:
                        n_subj = item.shape[0]
                        for s in range(n_subj):
                            subj = item[s]
                            # subj likely (n_samples, n_channels) -> transpose
                            if subj.ndim != 2:
                                continue
                            # ensure channels x time
                            if subj.shape[1] == N_CHANNELS:
                                eeg = subj.T
                            elif subj.shape[0] == N_CHANNELS:
                                eeg = subj
                            else:
                                # ambiguous: transpose to channels x time if needed
                                eeg = subj.T
                            if not np.isfinite(eeg).all():
                                continue
                            all_subj.append((eeg.astype(np.float32), label))
                    elif item.ndim == 2:
                        subj = item
                        if subj.shape[1] == N_CHANNELS:
                            eeg = subj.T
                        elif subj.shape[0] == N_CHANNELS:
                            eeg = subj
                        else:
                            eeg = subj.T
                        if not np.isfinite(eeg).all():
                            continue
                        all_subj.append((eeg.astype(np.float32), label))

            # direct ndarray stored at top-level
            elif v.ndim == 3:
                for s in range(v.shape[0]):
                    subj = v[s]
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
                    all_subj.append((eeg.astype(np.float32), label))
            elif v.ndim == 2:
                subj = v
                if subj.shape[1] == N_CHANNELS:
                    eeg = subj.T
                elif subj.shape[0] == N_CHANNELS:
                    eeg = subj
                else:
                    eeg = subj.T
                if not np.isfinite(eeg).all():
                    continue
                all_subj.append((eeg.astype(np.float32), label))
    return all_subj


# ---------- 2. PRE-PROCESS → 4×8×8 PSD MAP ----------
def preprocess_to_map(eeg_arr):
    """eeg_arr: (channels, time) → (4, 8, 8) interpolated PSD map

    Notes:
    - This function is robust to having only a few channels: it falls back
      to synthetic positions if standard montage lookup fails and fills
      missing grid values with nearest interpolation.
    """
    info = mne.create_info(ch_names=[f"Ch{i}" for i in range(N_CHANNELS)], sfreq=FS, ch_types=["eeg"] * N_CHANNELS)
    raw = mne.io.RawArray(eeg_arr, info)
    raw.filter(l_freq=0.5, h_freq=50., fir_design="firwin")
    raw.notch_filter(50.)

    # try adding a montage; if it fails, we'll create synthetic positions
    try:
        raw.set_montage("standard_1005")
        montage = mne.channels.make_standard_montage("standard_1005")
        ch_pos_map = montage.get_positions()["ch_pos"]
        pos = []
        for ch in raw.ch_names:
            p = ch_pos_map.get(ch)
            if p is None:
                raise KeyError
            pos.append(p)
        pos = np.array(pos)  # (n_ch, 3)
    except Exception:
        # Fallback: evenly space channels on a circle in the XY plane
        angles = np.linspace(0, 2 * np.pi, N_CHANNELS, endpoint=False)
        pos = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(N_CHANNELS)])

    # simple EOG projection attempt (ignore if fails)
    try:
        projs = mne.preprocessing.compute_proj_eog(raw, n_eeg=2)
        if projs:
            raw.add_proj(projs[0]).apply_proj()
    except Exception:
        pass

    clean = raw.get_data()  # (channels, time)

    # Welch bands
    freqs, psd = welch(clean, fs=FS, nperseg=N_PER_SEG, axis=-1)
    band_power = []
    for low, high in F_BANDS.values():
        idx = np.logical_and(freqs >= low, freqs <= high)
        # mean power in band for each channel
        band_power.append(psd[:, idx].mean(axis=-1))
    bp = np.array(band_power)  # (4, channels)

    # interpolate to 8×8 grid
    x_grid = np.linspace(pos[:, 0].min(), pos[:, 0].max(), 8)
    y_grid = np.linspace(pos[:, 1].min(), pos[:, 1].max(), 8)
    xx, yy = np.meshgrid(x_grid, y_grid)
    map_8x8 = np.zeros((4, 8, 8))
    for b in range(4):
        # if too few points for linear interpolation, use nearest
        try:
            grid_lin = griddata(pos[:, :2], bp[b], (xx, yy), method="linear")
            if np.isnan(grid_lin).any():
                grid_nearest = griddata(pos[:, :2], bp[b], (xx, yy), method="nearest")
                grid_lin = np.where(np.isnan(grid_lin), grid_nearest, grid_lin)
        except Exception:
            # fall back to nearest if linear fails (e.g., not enough points for Delaunay)
            grid_lin = griddata(pos[:, :2], bp[b], (xx, yy), method="nearest")
        map_8x8[b] = grid_lin
    return map_8x8  # (4, 8, 8)


# ---------- 3. CNN ENCODER ----------
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


# ---------- 4. TRAIN + SAVE ----------
def train_and_save(weights_dir="model"):
    Path(weights_dir).mkdir(exist_ok=True)
    # use absolute paths so script works from any directory
    script_dir = Path(__file__).parent
    mat_files = [
        script_dir / "data" / "FC.mat",
        script_dir / "data" / "MC.mat",
        script_dir / "data" / "FADHD.mat",
        script_dir / "data" / "MADHD.mat"
    ]
    mat_files = [str(p) for p in mat_files]
    all_subj = load_all_subjects(mat_files)  # list of (eeg, label)

    # ---- K-fold CV (5-fold) ----
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs = []

    # ---- build train + val splits ----
    X_full = []   # list of 4×8×8 maps
    y_full = []   # labels
    for eeg, label in all_subj:
        map_8x8 = preprocess_to_map(eeg)
        X_full.append(map_8x8)
        y_full.append(label)
    X_full = np.array(X_full)  # (n_subj, 4, 8, 8)
    y_full = np.array(y_full)  # (n_subj,)

    # ---- train 5-fold ----
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_full)):
        print(f"\n====== FOLD {fold+1}/5 ======")
        X_train, X_val = X_full[train_idx], X_full[val_idx]
        y_train, y_val = y_full[train_idx], y_full[val_idx]

        # ---- build PyTorch dataset ----
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_val_t   = torch.tensor(X_val, dtype=torch.float32)
        y_val_t   = torch.tensor(y_val, dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        loader  = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        # ---- model & optimizer ----
        cnn = EncoderCNN().to(DEVICE)
        optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        # ---- train epochs ----
        for epoch in range(15):
            cnn.train()
            for xb, yb in tqdm.tqdm(loader, desc=f"Epoch {epoch+1}"):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                out = cnn(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

            # ---- validate ----
            cnn.eval()
            with torch.no_grad():
                logits = cnn(X_val_t.to(DEVICE))
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                auc = skm.roc_auc_score(y_val, probs)
                print(f"  Val AUC: {auc:.3f}")
            fold_aucs.append(auc)

    print(f"\nMean AUC: {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}")

    # ---- retrain on FULL data ----
    print("\nRetraining on FULL data …")
    X_full_t = torch.tensor(X_full, dtype=torch.float32)
    full_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_full_t, torch.tensor(y_full, dtype=torch.long)),
        batch_size=32, shuffle=True)

    final_cnn = EncoderCNN().to(DEVICE)
    optimizer = torch.optim.Adam(final_cnn.parameters(), lr=1e-3)
    for epoch in range(15):
        final_cnn.train()
        for xb, yb in tqdm.tqdm(full_loader, desc=f"Full epoch {epoch+1}"):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = final_cnn(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    # ---- extract 16-D embeddings for prototypes ----
    final_cnn.eval()
    embeddings = []
    with torch.no_grad():
        for xb in torch.utils.data.DataLoader(torch.tensor(X_full, dtype=torch.float32), batch_size=32):
            z = final_cnn.embed(xb.to(DEVICE)).cpu().numpy()
            embeddings.append(z)
    embeddings = np.vstack(embeddings)  # (n_subj, embed_dim)

    # scale embeddings and compute class prototypes
    scaler = StandardScaler().fit(embeddings)
    embeddings_scaled = scaler.transform(embeddings)
    ctl_emb = embeddings_scaled[y_full == 0].mean(axis=0)
    adhd_emb = embeddings_scaled[y_full == 1].mean(axis=0)

    # ---- save ----
    torch.save(final_cnn.state_dict(), f"{weights_dir}/cnn_frozen.pth")
    joblib.dump([ctl_emb, adhd_emb], f"{weights_dir}/softmax_weights.pkl")
    joblib.dump(scaler, f"{weights_dir}/scaler_16d.pkl")
    print("✅ Real weights saved:")
    print(f"  {weights_dir}/cnn_frozen.pth")
    print(f"  {weights_dir}/softmax_weights.pt")
    print(f"  {weights_dir}/scaler_16d.pkl")


# ---------- 5. RUN NOW ----------
if __name__ == "__main__":
    train_and_save()