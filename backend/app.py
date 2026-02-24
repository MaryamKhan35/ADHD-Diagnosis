import base64
import csv
import io
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import bcrypt
import joblib
import jwt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import mne
import numpy as np
import torch
import mysql.connector
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from scipy.interpolate import griddata
from scipy.io import loadmat, savemat
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

# JWT and Auth Config
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440

# Database Config
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',  # Empty password - update if needed
    'database': 'eeg_adhd_detection'
}

# Pydantic Models
class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    first_name: str
    last_name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    phone: Optional[str] = None
    medical_facility: Optional[str] = None

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    patient_id: int
    email: str
    first_name: str
    last_name: str

class UserResponse(BaseModel):
    patient_id: int
    email: str
    first_name: str
    last_name: str
    age: Optional[int]
    gender: Optional[str]
    phone: Optional[str]
    medical_facility: Optional[str]

# Database helper functions
def get_db_connection():
    """Create and return a database connection."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as err:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {err}")

def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    """Verify and decode a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        patient_id: str = payload.get("sub")
        if patient_id is None:
            raise HTTPException(status_code=401, detail="Invalid token - no subject")
        return int(patient_id)  # Convert back to integer
    except jwt.ExpiredSignatureError as e:
        raise HTTPException(status_code=401, detail=f"Token expired: {str(e)}")
    except (jwt.InvalidTokenError, jwt.DecodeError) as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token error: {str(e)}")

app = FastAPI(title="EEG ADHD Backend")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174", "http://127.0.0.1:5174", "http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== AUTHENTICATION ENDPOINTS ====================

@app.post('/api/auth/signup', response_model=TokenResponse)
async def signup(request: SignupRequest):
    """Register a new user."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Check if email already exists
        cursor.execute("SELECT patient_id FROM PATIENT WHERE email = %s", (request.email,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Email already registered")

        # Hash password
        password_hash = hash_password(request.password)

        # Insert new patient
        insert_query = """
        INSERT INTO PATIENT (email, password_hash, first_name, last_name, age, gender, phone, medical_facility)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (
            request.email,
            password_hash,
            request.first_name,
            request.last_name,
            request.age,
            request.gender,
            request.phone,
            request.medical_facility
        ))
        conn.commit()
        patient_id = cursor.lastrowid

        # Create JWT token
        access_token = create_access_token(
            data={"sub": str(patient_id), "email": request.email},
            expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )

        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            patient_id=patient_id,
            email=request.email,
            first_name=request.first_name,
            last_name=request.last_name
        )

    except mysql.connector.Error as err:
        raise HTTPException(status_code=500, detail=f"Database error: {err}")
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


@app.post('/api/auth/login', response_model=TokenResponse)
async def login(request: LoginRequest):
    """Login an existing user."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Get user by email
        cursor.execute("SELECT patient_id, password_hash, first_name, last_name FROM PATIENT WHERE email = %s", (request.email,))
        user = cursor.fetchone()

        if not user or not verify_password(request.password, user['password_hash']):
            raise HTTPException(status_code=401, detail="Invalid email or password")

        # Create JWT token
        access_token = create_access_token(
            data={"sub": str(user['patient_id']), "email": request.email},
            expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )

        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            patient_id=user['patient_id'],
            email=request.email,
            first_name=user['first_name'],
            last_name=user['last_name']
        )

    except mysql.connector.Error as err:
        raise HTTPException(status_code=500, detail=f"Database error: {err}")
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


@app.get('/api/auth/user')
async def get_user(token: str = None):
    """Get current user info."""
    if not token:
        # Try to get from header
        raise HTTPException(status_code=401, detail="Token required")
    
    try:
        patient_id = verify_token(token)
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("SELECT patient_id, email, first_name, last_name, age, gender, phone, medical_facility FROM PATIENT WHERE patient_id = %s", (patient_id,))
        user = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return UserResponse(**user)
    except Exception as err:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.post('/api/auth/logout')
async def logout():
    """Logout endpoint."""
    # JWT tokens are stateless, so we just return success
    # Client should clear localStorage
    return {"message": "Logged out successfully"}


@app.get('/api/test/debug')
async def debug_test(authorization: str = Header(None, alias="Authorization")):
    """Debug endpoint to test token verification."""
    return {
        "received_header": authorization,
        "header_type": type(authorization).__name__
    }


# ==================== PATIENT ENDPOINTS ====================

def get_patient_id_from_header(authorization: str = Header(None, alias="Authorization")):
    """Extract and verify patient ID from Authorization header."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    try:
        parts = authorization.split()
        if len(parts) != 2 or parts[0].lower() != 'bearer':
            raise HTTPException(status_code=401, detail="Invalid authorization header")
        
        token = parts[1]
        return verify_token(token)
    except HTTPException:
        raise
    except Exception as err:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.get('/api/patient/profile')
async def get_profile(patient_id: int = Depends(get_patient_id_from_header)):
    """Get current patient's profile information."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute(
            "SELECT patient_id, email, first_name, last_name, age, gender, phone, medical_facility FROM PATIENT WHERE patient_id = %s",
            (patient_id,)
        )
        user = cursor.fetchone()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return UserResponse(**user)
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


@app.get('/api/patient/diagnosis-history')
async def get_diagnosis_history(patient_id: int = Depends(get_patient_id_from_header)):
    """Get patient's EEG diagnosis history - FAST endpoint."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Fast direct query without view
        cursor.execute("""
            SELECT 
                eu.upload_id,
                eu.filename,
                eu.created_at AS upload_date,
                eu.status,
                pr.classification,
                pr.adhd_probability,
                pr.confidence,
                mr.risk_level,
                mr.pdf_path
            FROM EEG_UPLOAD eu
            LEFT JOIN PREDICTION_RESULT pr ON eu.upload_id = pr.upload_id
            LEFT JOIN MEDICAL_REPORT mr ON pr.result_id = mr.result_id
            WHERE eu.patient_id = %s
            ORDER BY eu.created_at DESC
            LIMIT 10
        """, (patient_id,))
        
        diagnoses = cursor.fetchall()
        
        # Filter out rows with all nulls
        diagnoses = [d for d in diagnoses if d['filename'] is not None]
        
        return {
            "recent_diagnoses": diagnoses,
            "total_diagnoses": len(diagnoses)
        }
    except Exception as err:
        # Return empty list if query fails instead of error
        return {
            "recent_diagnoses": [],
            "total_diagnoses": 0
        }
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


# ==================== EEG PROCESSING ENDPOINTS ====================

@app.post('/api/patient/upload-and-predict')
async def upload_and_predict_authenticated(file: UploadFile = File(...), patient_id: int = Depends(get_patient_id_from_header)):
    """Upload EEG file and run prediction - SAVES RESULTS TO DATABASE.
    
    Authenticated endpoint that:
    1. Accepts .mat or .csv EEG files
    2. Runs preprocessing and prediction
    3. Saves results to database
    4. Returns prediction with upload ID
    """
    conn = None
    try:
        contents = await file.read()
        filename = file.filename
        
        # Process file (converts CSV to MAT if needed)
        data, file_format = process_upload_file(contents, filename)
        
        eeg = find_first_eeg_in_mat(data)
        if eeg is None:
            return {"error": "No valid EEG array found in file."}

        map8 = preprocess_to_map(eeg.astype(np.float32))

        # Load model + prototypes
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
            x = torch.tensor(map8, dtype=torch.float32).unsqueeze(0)
            emb = cnn.embed(x.to(DEVICE)).cpu().numpy()

        emb_scaled = scaler.transform(emb)
        ctl_emb, adhd_emb = prot
        d_ctl = np.linalg.norm(emb_scaled - ctl_emb, axis=1)
        d_adhd = np.linalg.norm(emb_scaled - adhd_emb, axis=1)
        scores = np.vstack([-d_ctl, -d_adhd]).T
        exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        prob_adhd = float(probs[0, 1])
        
        # Determine classification
        classification = 'ADHD' if prob_adhd > 0.5 else 'Control'
        confidence = prob_adhd if prob_adhd > 0.5 else (1 - prob_adhd)

        # Generate brain map image
        img_b64 = map_to_png_b64(map8)

        # Save to database
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # 1. Save to EEG_UPLOAD table
        cursor.execute("""
            INSERT INTO EEG_UPLOAD (patient_id, filename, file_size, status, created_at)
            VALUES (%s, %s, %s, %s, NOW())
        """, (patient_id, filename, len(contents), 'completed'))
        conn.commit()
        upload_id = cursor.lastrowid

        # 2. Save to PREDICTION_RESULT table
        cursor.execute("""
            INSERT INTO PREDICTION_RESULT (upload_id, model_id, classification, adhd_probability, confidence, created_at)
            VALUES (%s, %s, %s, %s, %s, NOW())
        """, (upload_id, 1, classification, prob_adhd, confidence))
        conn.commit()

        return {
            "success": True,
            "upload_id": upload_id,
            "filename": filename,
            "classification": classification,
            "probability_adhd": prob_adhd,
            "confidence": confidence,
            "map_png_base64": img_b64,
            "message": f"Diagnosis saved: {classification}"
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": str(e)}
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()



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
    is_csv: if True, skip filtering (data already processed)
    returns: np.array shape (4, 8, 8)
    """
    # normalize input
    eeg_arr = np.asarray(eeg_arr)
    print(f"Input shape: {eeg_arr.shape}")
    
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
    
    # Apply filters with verbose=False to avoid hanging
    raw.filter(l_freq=0.5, h_freq=50., fir_design="firwin", verbose=False)
    raw.notch_filter(50., verbose=False)
    clean = raw.get_data()

    # ensure clean has shape (channels, time)
    if clean.ndim == 1:
        clean = clean[np.newaxis, :]
    if clean.ndim != 2:
        raise ValueError(f"clean data has unexpected ndim={clean.ndim}, shape={clean.shape}")
    if clean.shape[0] != N_CHANNELS and clean.shape[1] == N_CHANNELS:
        clean = clean.T

    print(f"After preprocessing: {clean.shape}")

    # Synthetic positions (since we only have 2 channels)
    angles = np.linspace(0, 2 * np.pi, N_CHANNELS, endpoint=False)
    pos = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(N_CHANNELS)])

    # Compute band power
    print("Computing band power...")
    band_power = []
    
    # Welch PSD computation
    try:
        nperseg = min(N_PER_SEG, clean.shape[1] // 2)
        freqs, psd = welch(clean, fs=FS, nperseg=nperseg, axis=-1)
    except Exception as e:
        raise RuntimeError(f"Welch PSD computation failed: {e}; clean.shape={clean.shape}")

    # psd expected shape (channels, n_freqs)
    if psd.ndim != 2:
        raise ValueError(f"psd has unexpected ndim={psd.ndim}, shape={getattr(psd,'shape',None)}")

    for low, high in F_BANDS.values():
        idx = np.logical_and(freqs >= low, freqs <= high)
        band_power.append(psd[:, idx].mean(axis=-1))
    
    bp = np.array(band_power)  # (4, channels)
    print(f"Band power shape: {bp.shape}, range: [{bp.min():.2f}, {bp.max():.2f}]")

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
    
    print(f"Final map shape: {map_8x8.shape}")
    return map_8x8


def map_to_png_b64(map_4x8x8: np.ndarray) -> str:
    """Convert 4x8x8 brain map to PNG base64 string."""
    try:
        # Create an RGB-like visualization by stacking normalized bands
        norm = (map_4x8x8 - np.nanmin(map_4x8x8)) / (np.nanmax(map_4x8x8) - np.nanmin(map_4x8x8) + 1e-9)
        # stack first 3 bands as rgb; if only 4 bands, ignore the fourth
        rgb = np.dstack([norm[0], norm[1], norm[2]])
        
        fig = plt.figure(figsize=(3, 3), dpi=100)
        plt.axis('off')
        plt.imshow(rgb, origin='lower')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('ascii')
        return img_b64
    except Exception as e:
        raise RuntimeError(f"Failed to generate brain map PNG: {e}")


def convert_csv_to_mat(csv_contents: bytes) -> np.ndarray:
    """Convert CSV file to proper .mat format (1, 11) cell array with F3-F4 channels.
    
    Expected CSV format: EEG channel columns + "Class" + "ID" columns
    Returns: (1, 11) cell array where each element is shape (n_samples, 2) with F3-F4 channels
    """
    # Parse CSV from bytes
    csv_str = csv_contents.decode('utf-8')
    csv_reader = csv.DictReader(io.StringIO(csv_str))
    rows = list(csv_reader)
    
    if not rows:
        raise ValueError("CSV file is empty or invalid")
    
    # Get the first row to check for required channels
    header = rows[0].keys()
    if 'F3' not in header or 'F4' not in header:
        raise ValueError(f"CSV must contain 'F3' and 'F4' columns. Found: {list(header)}")
    
    # Extract F3 and F4 channels from all rows
    channels_list = []
    for row in rows:
        try:
            f3 = float(row['F3'])
            f4 = float(row['F4'])
            channels_list.append([f3, f4])
        except (ValueError, KeyError) as e:
            raise ValueError(f"Error parsing F3/F4 values in row: {e}")
    
    channels = np.array(channels_list, dtype=np.float64)
    
    # Normalize using z-score and scale to match FADHD standard (std=17.56)
    mean_val = channels.mean()
    std_val = channels.std()
    if std_val == 0:
        raise ValueError("Data has zero standard deviation; cannot normalize")
    
    channels_normalized = (channels - mean_val) / std_val * 17.56
    
    # Create the exact format: (1, 11) cell array with float64 elements
    mat_data = np.empty((1, 11), dtype=object)
    for i in range(11):
        mat_data[0, i] = channels_normalized
    
    return mat_data


def process_upload_file(file_contents: bytes, filename: str) -> tuple:
    """Process uploaded file (CSV or MAT) and return MAT-format data.
    
    If CSV: converts to proper .mat format
    If MAT: returns as-is
    
    Returns: (mat_dict, original_format)
    """
    filename_lower = filename.lower()
    
    if filename_lower.endswith('.csv'):
        # Convert CSV to MAT format
        mat_data = convert_csv_to_mat(file_contents)
        mat_dict = {'data': mat_data}
        return mat_dict, 'csv'
    
    elif filename_lower.endswith('.mat'):
        # Load MAT file directly
        mat_dict = loadmat_from_bytes(file_contents)
        return mat_dict, 'mat'
    
    else:
        raise ValueError(f"Unsupported file format: {filename}. Please upload .csv or .mat files.")





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
    """Accepts a .mat or .csv EEG file and returns a preview PSD brainmap (base64 PNG).
    
    If CSV is uploaded, automatically converts it to proper .mat format with F3-F4 channels.
    """
    try:
        contents = await file.read()
        filename = file.filename
        
        # Process file (converts CSV to MAT if needed)
        data, file_format = process_upload_file(contents, filename)
        
        eeg = find_first_eeg_in_mat(data)
        if eeg is None:
            return {"error": "No valid EEG array found in file."}
        
        map8 = preprocess_to_map(eeg.astype(np.float32))
        img_b64 = map_to_png_b64(map8)
        return {
            "map_shape": map8.shape, 
            "map_png_base64": img_b64,
            "input_format": file_format,
            "conversion_note": "CSV automatically converted to MAT format with F3-F4 channels" if file_format == 'csv' else None
        }
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"Error in preprocess: {tb}")
        return {"error": str(e), "traceback": tb}


@app.post('/predict')
async def upload_and_predict(file: UploadFile = File(...)) -> Dict:
    """Run preprocessing + model embedding and return ADHD probability.
    
    Accepts .mat or .csv EEG files.
    If CSV is uploaded, automatically converts it to proper .mat format with F3-F4 channels.

    Requires trained files in `model/`:
      - cnn_frozen.pth
      - softmax_weights.pkl
      - scaler_16d.pkl
    """
    try:
        contents = await file.read()
        filename = file.filename
        
        # Process file (converts CSV to MAT if needed)
        data, file_format = process_upload_file(contents, filename)
        
        eeg = find_first_eeg_in_mat(data)
        if eeg is None:
            return {"error": "No valid EEG array found in file. Expected a 2D array with shape (samples, 2) or (2, samples), or a 3D array with shape (subjects, samples, 2)."}

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
        return {
            "probability_adhd": prob_adhd, 
            "map_shape": map8.shape, 
            "map_png_base64": img_b64,
            "input_format": file_format,
            "conversion_note": "CSV automatically converted to MAT format with F3-F4 channels" if file_format == 'csv' else None
        }
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
    """Extract first valid EEG array with 2 channels from MAT file."""
    for k, v in mat.items():
        if k.startswith('__'):
            continue
        if not isinstance(v, np.ndarray):
            continue

        # Case: MATLAB cell array (object)
        if v.dtype == np.object_:
            for item in v.flat:
                if not isinstance(item, np.ndarray):
                    continue
                # 3D array: (n_subjects, samples, 2) or (n_subjects, 2, samples)
                if item.ndim == 3 and item.shape[0] >= 1:
                    subj = item[0]
                    if subj.ndim == 2:
                        # Check which dimension has 2 channels
                        if subj.shape[0] == N_CHANNELS:
                            arr = subj
                        elif subj.shape[1] == N_CHANNELS:
                            arr = subj.T
                        else:
                            continue
                        
                        if arr.shape[0] == N_CHANNELS and np.isfinite(arr).all():
                            return arr
                # 2D array: (samples, 2) or (2, samples)
                if item.ndim == 2 and np.issubdtype(item.dtype, np.number):
                    if item.shape[0] == N_CHANNELS:
                        arr = item
                    elif item.shape[1] == N_CHANNELS:
                        arr = item.T
                    else:
                        continue
                    
                    if arr.shape[0] == N_CHANNELS and np.isfinite(arr).all():
                        return arr

        # Case: direct 3D ndarray
        if v.ndim == 3 and v.shape[0] >= 1:
            subj = v[0]
            if subj.ndim == 2 and np.issubdtype(subj.dtype, np.number):
                if subj.shape[0] == N_CHANNELS:
                    arr = subj
                elif subj.shape[1] == N_CHANNELS:
                    arr = subj.T
                else:
                    continue
                
                if arr.shape[0] == N_CHANNELS and np.isfinite(arr).all():
                    return arr

        # Case: direct 2D ndarray
        if v.ndim == 2 and np.issubdtype(v.dtype, np.number):
            if v.shape[0] == N_CHANNELS:
                arr = v
            elif v.shape[1] == N_CHANNELS:
                arr = v.T
            else:
                continue
            
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
