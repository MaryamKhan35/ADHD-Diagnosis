"""
train_with_test_split.py
Train CNN with proper train/test split: 80% train (with 5-fold CV), 20% test (holdout)
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
from sklearn.model_selection import KFold, train_test_split
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
N_CHANNELS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- 1. LOAD ALL SUBJECTS ----------
def load_all_subjects(mat_paths):
    """Returns list of (eeg, label)  eeg=(channels, time)  label=0/1"""
    all_subj = []
    for path, label in zip(mat_paths, [0, 0, 1, 1]):  # FC=0, MC=0, FADHD=1, MADHD=1
        mat = sio.loadmat(path)
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


# ---------- 2. PRE-PROCESS -> 4x8x8 PSD MAP ----------
def preprocess_to_map(eeg_arr):
    """eeg_arr: (channels, time) -> (4, 8, 8) interpolated PSD map"""
    info = mne.create_info(ch_names=[f"Ch{i}" for i in range(N_CHANNELS)], sfreq=FS, ch_types=["eeg"] * N_CHANNELS)
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

    try:
        projs = mne.preprocessing.compute_proj_eog(raw, n_eeg=2)
        if projs:
            raw.add_proj(projs[0]).apply_proj()
    except Exception:
        pass

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
    script_dir = Path(__file__).parent
    mat_files = [
        script_dir / "data" / "FC.mat",
        script_dir / "data" / "MC.mat",
        script_dir / "data" / "FADHD.mat",
        script_dir / "data" / "MADHD.mat"
    ]
    mat_files = [str(p) for p in mat_files]
    all_subj = load_all_subjects(mat_files)

    print("\n" + "="*70)
    print("TRAIN/TEST SPLIT: 80% TRAIN, 20% TEST")
    print("="*70)
    print("Loading and preprocessing all subjects...")
    
    # Preprocess all to maps first
    X_all = []
    y_all = []
    for i, (eeg, label) in enumerate(all_subj):
        if (i + 1) % 100 == 0:
            print("  Processed {}/{}".format(i + 1, len(all_subj)))
        map_8x8 = preprocess_to_map(eeg)
        X_all.append(map_8x8)
        y_all.append(label)

    X_all = np.array(X_all)
    y_all = np.array(y_all)

    print("Total samples: {}".format(len(X_all)))
    print("  Control: {}, ADHD: {}".format(sum(y_all==0), sum(y_all==1)))

    # Split: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    print("\nTRAIN SET: {} samples".format(len(X_train)))
    print("  Control: {}, ADHD: {}".format(sum(y_train==0), sum(y_train==1)))
    print("TEST SET:  {} samples".format(len(X_test)))
    print("  Control: {}, ADHD: {}".format(sum(y_test==0), sum(y_test==1)))

    # ---- K-fold CV on TRAIN set (5-fold) ----
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs = []

    print("\n" + "="*70)
    print("5-FOLD CROSS-VALIDATION ON TRAINING SET")
    print("="*70)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print("\n====== FOLD {}/5 ======".format(fold+1))
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

        X_fold_train_t = torch.tensor(X_fold_train, dtype=torch.float32)
        y_fold_train_t = torch.tensor(y_fold_train, dtype=torch.long)
        X_fold_val_t   = torch.tensor(X_fold_val, dtype=torch.float32)
        y_fold_val_t   = torch.tensor(y_fold_val, dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(X_fold_train_t, y_fold_train_t)
        loader  = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        cnn = EncoderCNN().to(DEVICE)
        optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(15):
            cnn.train()
            for xb, yb in tqdm.tqdm(loader, desc="Epoch {}".format(epoch+1)):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                out = cnn(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

            cnn.eval()
            with torch.no_grad():
                logits = cnn(X_fold_val_t.to(DEVICE))
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                auc = skm.roc_auc_score(y_fold_val, probs)
                print("  Val AUC: {:.4f}".format(auc))
            fold_aucs.append(auc)

    print("\nMean 5-Fold AUC: {:.4f} +/- {:.4f}".format(np.mean(fold_aucs), np.std(fold_aucs)))

    # ---- Retrain on FULL TRAIN set ----
    print("\n" + "="*70)
    print("RETRAINING ON FULL TRAINING SET")
    print("="*70)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    full_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_t, torch.tensor(y_train, dtype=torch.long)),
        batch_size=32, shuffle=True)

    final_cnn = EncoderCNN().to(DEVICE)
    optimizer = torch.optim.Adam(final_cnn.parameters(), lr=1e-3)
    for epoch in range(15):
        final_cnn.train()
        for xb, yb in tqdm.tqdm(full_loader, desc="Full epoch {}".format(epoch+1)):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = final_cnn(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    # ---- Extract embeddings for prototypes ----
    final_cnn.eval()
    embeddings = []
    with torch.no_grad():
        for xb in torch.utils.data.DataLoader(torch.tensor(X_train, dtype=torch.float32), batch_size=32):
            z = final_cnn.embed(xb.to(DEVICE)).cpu().numpy()
            embeddings.append(z)
    embeddings = np.vstack(embeddings)

    scaler = StandardScaler().fit(embeddings)
    embeddings_scaled = scaler.transform(embeddings)
    ctl_emb = embeddings_scaled[y_train == 0].mean(axis=0)
    adhd_emb = embeddings_scaled[y_train == 1].mean(axis=0)

    # ---- Save ----
    torch.save(final_cnn.state_dict(), "{}/cnn_frozen.pth".format(weights_dir))
    joblib.dump([ctl_emb, adhd_emb], "{}/softmax_weights.pkl".format(weights_dir))
    joblib.dump(scaler, "{}/scaler_16d.pkl".format(weights_dir))
    print("\nWeights saved to: {}".format(weights_dir))

    # ---- EVALUATE ON TEST SET ----
    print("\n" + "="*70)
    print("EVALUATING ON HOLDOUT TEST SET")
    print("="*70)

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    test_loader = torch.utils.data.DataLoader(X_test_t, batch_size=32, shuffle=False)

    test_embeddings = []
    with torch.no_grad():
        for xb in test_loader:
            z = final_cnn.embed(xb.to(DEVICE)).cpu().numpy()
            test_embeddings.append(z)
    test_embeddings = np.vstack(test_embeddings)

    test_embeddings_scaled = scaler.transform(test_embeddings)
    d_ctl = np.linalg.norm(test_embeddings_scaled - ctl_emb, axis=1)
    d_adhd = np.linalg.norm(test_embeddings_scaled - adhd_emb, axis=1)

    scores = np.vstack([-d_ctl, -d_adhd]).T
    exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    prob_adhd_test = probs[:, 1]
    y_pred_test = (prob_adhd_test > 0.5).astype(int)

    # Calculate metrics
    acc_test = skm.accuracy_score(y_test, y_pred_test)
    precision_test = skm.precision_score(y_test, y_pred_test, zero_division=0)
    recall_test = skm.recall_score(y_test, y_pred_test, zero_division=0)
    f1_test = skm.f1_score(y_test, y_pred_test, zero_division=0)
    roc_auc_test = skm.roc_auc_score(y_test, prob_adhd_test)

    cm_test = skm.confusion_matrix(y_test, y_pred_test)
    tn_test, fp_test, fn_test, tp_test = cm_test.ravel()

    print("\nTEST SET RESULTS:")
    print("  Accuracy:  {:.4f} ({:.2f}%)".format(acc_test, acc_test*100))
    print("  Precision: {:.4f}".format(precision_test))
    print("  Recall:    {:.4f}".format(recall_test))
    print("  F1-Score:  {:.4f}".format(f1_test))
    print("  ROC-AUC:   {:.4f}".format(roc_auc_test))

    print("\nConfusion Matrix (Test Set):")
    print("                Predicted")
    print("            Control  ADHD")
    print("  Control      {}    {}".format(tn_test, fp_test))
    print("  ADHD         {}    {}".format(fn_test, tp_test))

    print("\n" + "="*70)
    print("SUMMARY:")
    print("  Train/Val (5-fold): {:.4f}".format(np.mean(fold_aucs)))
    print("  Test Set Accuracy:  {:.4f} ({:.2f}%)".format(acc_test, acc_test*100))
    print("="*70)


if __name__ == "__main__":
    train_and_save()
