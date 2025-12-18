import base64
import io
import os
from pathlib import Path
from typing import Dict

import joblib
import matplotlib.pyplot as plt
import mne
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from scipy.interpolate import griddata
from scipy.io import loadmat
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler

# Config
FS = 256
N_PER_SEG = 512
F_BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 50)}
N_CHANNELS = 2
MODEL_DIR = Path(__file__).parent.parent / "model"
MODEL_DIR = Path(MODEL_DIR)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="EEG ADHD Backend")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Local copy of encoder (must match architecture used in training)
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
    """Minimal preprocessing + PSD band interpolation to 4x8x8 map.

    eeg_arr expected shape: (channels, time)
    returns: np.array shape (4, 8, 8)
    """
    # normalize input
    eeg_arr = np.asarray(eeg_arr)
    # expected (channels, time)
    if eeg_arr.ndim == 1:
        raise ValueError(f"eeg_arr is 1-D with length {eeg_arr.shape[0]}; expected 2-D (channels, time)")
    if eeg_arr.ndim == 3:
        # sometimes a subject dimension sneaks in: pick first subject
        eeg_arr = eeg_arr[0]

    if eeg_arr.shape[0] != N_CHANNELS and eeg_arr.shape[1] == N_CHANNELS:
        eeg_arr = eeg_arr.T

    if eeg_arr.shape[0] != N_CHANNELS:
        raise ValueError(f"Expected {N_CHANNELS} channels as first dimension, got shape {eeg_arr.shape}")

    # build minimal RawArray
    info = mne.create_info(ch_names=[f"Ch{i}" for i in range(N_CHANNELS)], sfreq=FS, ch_types=["eeg"] * N_CHANNELS)
    raw = mne.io.RawArray(eeg_arr, info)
    raw.filter(l_freq=0.5, h_freq=50., fir_design="firwin")
    raw.notch_filter(50.)

    # try montage -> fallback synthetic positions
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
        pos = np.array(pos)
    except Exception:
        angles = np.linspace(0, 2 * np.pi, N_CHANNELS, endpoint=False)
        pos = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(N_CHANNELS)])

    clean = raw.get_data()
    # ensure clean has shape (channels, time)
    if clean.ndim == 1:
        clean = clean[np.newaxis, :]
    if clean.ndim != 2:
        raise ValueError(f"clean data has unexpected ndim={clean.ndim}, shape={clean.shape}")
    if clean.shape[0] != N_CHANNELS and clean.shape[1] == N_CHANNELS:
        clean = clean.T

    # compute PSD per channel
    try:
        freqs, psd = welch(clean, fs=FS, nperseg=N_PER_SEG, axis=-1)
    except Exception as e:
        raise RuntimeError(f"Welch PSD computation failed: {e}; clean.shape={clean.shape}")

    # psd expected shape (channels, n_freqs)
    if psd.ndim != 2:
        # if user passed 1D, provide helpful error
        raise ValueError(f"psd has unexpected ndim={psd.ndim}, shape={getattr(psd,'shape',None)}")

    band_power = []
    for low, high in F_BANDS.values():
        idx = np.logical_and(freqs >= low, freqs <= high)
        band_power.append(psd[:, idx].mean(axis=-1))
    bp = np.array(band_power)  # (4, channels)

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
    return map_8x8


def map_to_png_b64(map_4x8x8: np.ndarray) -> str:
    # Create an RGB-like visualization by stacking normalized bands
    norm = (map_4x8x8 - np.nanmin(map_4x8x8)) / (np.nanmax(map_4x8x8) - np.nanmin(map_4x8x8) + 1e-9)
    # stack first 3 bands as rgb; if only 4 bands, ignore the fourth
    rgb = np.dstack([norm[0], norm[1], norm[2]])
    plt.figure(figsize=(3, 3), dpi=100)
    plt.axis('off')
    plt.imshow(rgb, origin='lower')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('ascii')
    return img_b64


def load_prototypes_and_scaler(model_dir: Path):
    prot_path = model_dir / "softmax_weights.pkl"
    scaler_path = model_dir / "scaler_16d.pkl"
    if not prot_path.exists() or not scaler_path.exists():
        return None, None
    prot = joblib.load(prot_path)
    scaler = joblib.load(scaler_path)
    return prot, scaler


@app.post('/preprocess')
async def upload_and_preprocess(file: UploadFile = File(...)) -> Dict:
    """Accepts a .mat EEG file and returns a preview PSD brainmap (base64 PNG)."""
    try:
        contents = await file.read()
        data = loadmat_from_bytes(contents)
        # try to find first cell containing a 2D EEG array
        eeg = find_first_eeg_in_mat(data)
        if eeg is None:
            return {"error": "No valid EEG array found in MAT file."}
        map8 = preprocess_to_map(eeg.astype(np.float32))
        img_b64 = map_to_png_b64(map8)
        return {"map_shape": map8.shape, "map_png_base64": img_b64}
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return {"error": str(e), "traceback": tb}


@app.post('/predict')
async def upload_and_predict(file: UploadFile = File(...)) -> Dict:
    """Run preprocessing + model embedding and return ADHD probability.

    Requires trained files in `model/`:
      - cnn_frozen.pth
      - softmax_weights.pkl
      - scaler_16d.pkl
    """
    try:
        contents = await file.read()
        data = loadmat_from_bytes(contents)
        eeg = find_first_eeg_in_mat(data)
        if eeg is None:
            return {"error": "No valid EEG array found in MAT file. Expected a 2D array with shape (samples, 2) or (2, samples), or a 3D array with shape (subjects, samples, 2)."}

        map8 = preprocess_to_map(eeg.astype(np.float32))

        # load model + prototypes
        prot, scaler = load_prototypes_and_scaler(MODEL_DIR)
        if prot is None or scaler is None:
            return {"error": "Model artifacts not found. Please run training first."}

        cnn = EncoderCNN().to(DEVICE)
        cnn_path = MODEL_DIR / "cnn_frozen.pth"
        if not cnn_path.exists():
            return {"error": "CNN weights not found. Please run training first."}
        state = torch.load(cnn_path, map_location=DEVICE)
        cnn.load_state_dict(state)
        cnn.eval()

        with torch.no_grad():
            x = torch.tensor(map8, dtype=torch.float32).unsqueeze(0)  # (1,4,8,8)
            emb = cnn.embed(x.to(DEVICE)).cpu().numpy()

        emb_scaled = scaler.transform(emb)  # (1,dim)
        ctl_emb, adhd_emb = prot
        # Euclidean distances
        d_ctl = np.linalg.norm(emb_scaled - ctl_emb, axis=1)
        d_adhd = np.linalg.norm(emb_scaled - adhd_emb, axis=1)
        # convert distances to probability via softmax-like
        scores = np.vstack([-d_ctl, -d_adhd]).T
        exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        prob_adhd = float(probs[0, 1])

        img_b64 = map_to_png_b64(map8)
        return {"probability_adhd": prob_adhd, "map_shape": map8.shape, "map_png_base64": img_b64}
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return {"error": str(e), "traceback": tb}


# --- helpers ---

def loadmat_from_bytes(b: bytes) -> Dict:
    # save to temp file and use scipy.io.loadmat
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as tf:
        tf.write(b)
        tmp = tf.name
    try:
        mat = loadmat(tmp)
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass
    return mat


def find_first_eeg_in_mat(mat: Dict):
    # Heuristics: handle several common layouts from your dataset
    # - cell arrays where each cell is an ndarray shaped (n_subjects, n_samples, n_channels)
    # - direct ndarrays shaped (n_subjects, n_samples, n_channels)
    # - 2D arrays shaped (n_samples, n_channels)
    for k, v in mat.items():
        if k.startswith('__'):
            continue
        if not isinstance(v, np.ndarray):
            continue

        # Case: MATLAB cell array (object) -> iterate contained items
        if v.dtype == np.object_:
            for item in v.flat:
                if not isinstance(item, np.ndarray):
                    continue
                # if item is 3D: (n_subjects, n_samples, n_channels)
                if item.ndim == 3 and item.shape[0] >= 1:
                    # return first subject found (transpose to channels x time)
                    subj = item[0]
                    if subj.ndim == 2:
                        # Check which dimension has N_CHANNELS (2)
                        if subj.shape[1] == N_CHANNELS:
                            # (samples, channels) -> transpose to (channels, samples)
                            arr = subj.T
                        elif subj.shape[0] == N_CHANNELS:
                            # Already (channels, samples)
                            arr = subj
                        else:
                            # Default: transpose
                            arr = subj.T
                        
                        # Validate: must have exactly N_CHANNELS channels
                        if arr.shape[0] == N_CHANNELS and np.isfinite(arr).all():
                            return arr
                # if item is 2D: assume (n_samples, n_channels)
                if item.ndim == 2 and np.issubdtype(item.dtype, np.number):
                    # Check which dimension has N_CHANNELS
                    if item.shape[1] == N_CHANNELS:
                        arr = item.T  # (samples, channels) -> (channels, samples)
                    elif item.shape[0] == N_CHANNELS:
                        arr = item  # Already (channels, samples)
                    else:
                        arr = item.T  # Default: transpose
                    
                    # Validate: must have exactly N_CHANNELS channels
                    if arr.shape[0] == N_CHANNELS and np.isfinite(arr).all():
                        return arr

        # Case: direct ndarray
        if v.ndim == 3 and v.shape[0] >= 1:
            subj = v[0]
            if subj.ndim == 2 and np.issubdtype(subj.dtype, np.number):
                # Check which dimension has N_CHANNELS
                if subj.shape[1] == N_CHANNELS:
                    arr = subj.T  # (samples, channels) -> (channels, samples)
                elif subj.shape[0] == N_CHANNELS:
                    arr = subj  # Already (channels, samples)
                else:
                    arr = subj.T  # Default: transpose
                
                # Validate: must have exactly N_CHANNELS channels
                if arr.shape[0] == N_CHANNELS and np.isfinite(arr).all():
                    return arr

        if v.ndim == 2 and np.issubdtype(v.dtype, np.number):
            # Check which dimension has N_CHANNELS
            if v.shape[1] == N_CHANNELS:
                arr = v.T  # (samples, channels) -> (channels, samples)
            elif v.shape[0] == N_CHANNELS:
                arr = v  # Already (channels, samples)
            else:
                arr = v.T  # Default: transpose
            
            # Validate: must have exactly N_CHANNELS channels
            if arr.shape[0] == N_CHANNELS and np.isfinite(arr).all():
                return arr

    return None


@app.post('/debug_mat')
async def debug_mat(file: UploadFile = File(...)) -> Dict:
    """Return keys and a short summary (dtype/shape) of entries in the uploaded .mat file.

    This helps identify where the EEG arrays live in custom MAT structures.
    """
    contents = await file.read()
    mat = loadmat_from_bytes(contents)
    summary = {}
    for k, v in mat.items():
        if k.startswith('__'):
            continue
        try:
            if isinstance(v, np.ndarray):
                if v.dtype == np.object_:
                    # summarize first few elements of the cell
                    elems = []
                    count = 0
                    for item in v.flat:
                        if count >= 6:
                            break
                        if isinstance(item, np.ndarray):
                            elems.append({
                                'type': 'ndarray',
                                'shape': item.shape,
                                'dtype': str(item.dtype)
                            })
                        else:
                            elems.append({'type': str(type(item))})
                        count += 1
                    summary[k] = {'kind': 'cell-array', 'size': v.shape, 'sample_elements': elems}
                else:
                    summary[k] = {'kind': 'ndarray', 'shape': v.shape, 'dtype': str(v.dtype)}
            else:
                summary[k] = {'kind': str(type(v))}
        except Exception as e:
            summary[k] = {'error': str(e)}
    return {'keys_summary': summary}
