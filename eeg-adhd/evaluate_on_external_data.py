"""
Comprehensive model evaluation on all available data formats
"""
import scipy.io as sio
import numpy as np
import torch
import joblib
from pathlib import Path
import sklearn.metrics as skm
import mne
from scipy.signal import welch
from scipy.interpolate import griddata
import warnings
import os

warnings.filterwarnings("ignore")
os.environ['MNE_LOGGING_LEVEL'] = 'ERROR'

FS = 256
N_PER_SEG = 512
F_BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 50)}
N_CHANNELS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        raise ValueError(f"EEG is 1-D")
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


def load_mat_file(mat_path, label):
    """Load EEG from .mat file"""
    subjects = []
    mat = sio.loadmat(str(mat_path))
    
    for k, v in mat.items():
        if k.startswith('__'):
            continue
        if not isinstance(v, np.ndarray):
            continue

        if v.dtype == object:
            for item in v.flat:
                if not isinstance(item, np.ndarray):
                    continue
                if item.ndim == 3:
                    for s in range(item.shape[0]):
                        subj = item[s]
                        if subj.shape[1] == N_CHANNELS:
                            eeg = subj.T
                        else:
                            eeg = subj
                        if np.isfinite(eeg).all():
                            subjects.append((eeg.astype(np.float32), label))
                elif item.ndim == 2:
                    subj = item
                    if subj.shape[1] == N_CHANNELS:
                        eeg = subj.T
                    else:
                        eeg = subj
                    if np.isfinite(eeg).all():
                        subjects.append((eeg.astype(np.float32), label))
        elif v.ndim == 3:
            for s in range(v.shape[0]):
                subj = v[s]
                if subj.shape[1] == N_CHANNELS:
                    eeg = subj.T
                else:
                    eeg = subj
                if np.isfinite(eeg).all():
                    subjects.append((eeg.astype(np.float32), label))
        elif v.ndim == 2:
            subj = v
            if subj.shape[1] == N_CHANNELS:
                eeg = subj.T
            else:
                eeg = subj
            if np.isfinite(eeg).all():
                subjects.append((eeg.astype(np.float32), label))
    
    return subjects


def evaluate_dataset(name, mat_files, model_dir):
    """Evaluate on a dataset from .mat files"""
    print(f"\n{'='*70}")
    print(f"EVALUATING: {name}")
    print(f"{'='*70}")
    
    # Load subjects
    all_subj = []
    for mat_path, label in mat_files:
        subj = load_mat_file(mat_path, label)
        all_subj.extend(subj)
        print(f"Loaded {len(subj)} subjects from {Path(mat_path).name}")
    
    print(f"Total subjects: {len(all_subj)}")
    
    if len(all_subj) == 0:
        print("No subjects loaded!")
        return
    
    # Preprocess
    print("Preprocessing...")
    X, y = [], []
    for i, (eeg, label) in enumerate(all_subj):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(all_subj)}")
        try:
            map_8x8 = preprocess_to_map(eeg)
            X.append(map_8x8)
            y.append(label)
        except Exception as e:
            continue
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Preprocessed: {len(X)} samples")
    print(f"  Control: {sum(y==0)}, ADHD: {sum(y==1)}")
    
    # Load model
    cnn_path = model_dir / "cnn_frozen.pth"
    scaler_path = model_dir / "scaler_16d.pkl"
    prot_path = model_dir / "softmax_weights.pkl"
    
    cnn = EncoderCNN().to(DEVICE)
    cnn.load_state_dict(torch.load(cnn_path, map_location=DEVICE))
    cnn.eval()
    
    scaler = joblib.load(scaler_path)
    ctl_emb, adhd_emb = joblib.load(prot_path)
    
    # Predict
    X_t = torch.tensor(X, dtype=torch.float32)
    embeddings = []
    
    with torch.no_grad():
        for batch_idx in range(0, len(X), 32):
            batch = X_t[batch_idx:batch_idx+32].to(DEVICE)
            z = cnn.embed(batch).cpu().numpy()
            embeddings.append(z)
    
    embeddings = np.vstack(embeddings)
    embeddings_scaled = scaler.transform(embeddings)
    
    d_ctl = np.linalg.norm(embeddings_scaled - ctl_emb, axis=1)
    d_adhd = np.linalg.norm(embeddings_scaled - adhd_emb, axis=1)
    
    scores = np.vstack([-d_ctl, -d_adhd]).T
    exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    prob_adhd = probs[:, 1]
    y_pred = (prob_adhd > 0.5).astype(int)
    
    # Metrics
    if len(np.unique(y)) > 1:
        acc = skm.accuracy_score(y, y_pred)
        precision = skm.precision_score(y, y_pred, zero_division=0)
        recall = skm.recall_score(y, y_pred, zero_division=0)
        f1 = skm.f1_score(y, y_pred, zero_division=0)
        roc_auc = skm.roc_auc_score(y, prob_adhd)
        
        cm = skm.confusion_matrix(y, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
        
        print(f"\nACCURACY:    {acc:.4f} ({acc*100:.2f}%)")
        print(f"PRECISION:   {precision:.4f}")
        print(f"RECALL:      {recall:.4f}")
        print(f"F1-SCORE:    {f1:.4f}")
        print(f"ROC-AUC:     {roc_auc:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"            Control  ADHD")
        print(f"  Control      {tn}    {fp}")
        print(f"  ADHD         {fn}    {tp}")
        print(f"\nClassification Report:")
        print(skm.classification_report(y, y_pred, target_names=['Control', 'ADHD']))
    else:
        print(f"Single class only. Predictions: {y_pred.sum()} ADHD, {len(y_pred) - y_pred.sum()} Control")
        print(f"Mean ADHD probability: {prob_adhd.mean():.4f}")


if __name__ == "__main__":
    data_dir = Path(__file__).parent / "data"
    model_dir = Path(__file__).parent.parent / "model"
    
    # Test on ADHD.mat (external test data)
    if (data_dir / "ADHD.mat").exists():
        evaluate_dataset(
            "ADHD.mat (External ADHD Data)",
            [(data_dir / "ADHD.mat", 1)],
            model_dir
        )
    
    # Test on ADHD2.mat
    if (data_dir / "ADHD2.mat").exists():
        evaluate_dataset(
            "ADHD2.mat (External ADHD Data)",
            [(data_dir / "ADHD2.mat", 1)],
            model_dir
        )
    
    # Test on ADHD3.mat
    if (data_dir / "ADHD3.mat").exists():
        evaluate_dataset(
            "ADHD3.mat (External ADHD Data)",
            [(data_dir / "ADHD3.mat", 1)],
            model_dir
        )
    
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")
