from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime, timedelta
from typing import Optional, List
import jwt
import bcrypt
import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv

load_dotenv()

# Database Configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "eeg_adhd_detection")
}

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

# Initialize FastAPI app
app = FastAPI(title="EEG ADHD Detection API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Database Connection ====================
def get_db_connection():
    """Get MySQL database connection"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Error as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

# ==================== Pydantic Models ====================
class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
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

class PatientProfile(BaseModel):
    patient_id: int
    email: str
    first_name: str
    last_name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    phone: Optional[str] = None
    medical_facility: Optional[str] = None
    role: str
    created_at: str
    last_login: Optional[str] = None

class DiagnosisRecord(BaseModel):
    upload_id: int
    filename: str
    upload_date: str
    status: str
    classification: Optional[str] = None
    adhd_probability: Optional[float] = None
    confidence: Optional[float] = None
    risk_level: Optional[str] = None
    pdf_path: Optional[str] = None

class DiagnosisHistory(BaseModel):
    patient_id: int
    total_diagnoses: int
    recent_diagnoses: List[DiagnosisRecord]

# ==================== Authentication Utilities ====================
def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(password: str, hash: str) -> bool:
    """Verify password against hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hash.encode('utf-8'))

def create_access_token(patient_id: int, email: str, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT token"""
    if expires_delta is None:
        expires_delta = timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    
    expire = datetime.utcnow() + expires_delta
    to_encode = {"patient_id": patient_id, "email": email, "exp": expire}
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> dict:
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        patient_id: int = payload.get("patient_id")
        email: str = payload.get("email")
        if patient_id is None or email is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"patient_id": patient_id, "email": email}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_patient(request: Request):
    """Get current logged-in patient from Authorization header"""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    
    token = auth_header.split(" ")[1]
    return verify_token(token)

# ==================== Authentication Routes ====================
@app.post("/api/auth/signup", response_model=TokenResponse)
async def signup(request: SignupRequest, db_request: Request):
    """Register new patient"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        # Check if email already exists
        cursor.execute("SELECT patient_id FROM PATIENT WHERE email = %s", (request.email,))
        existing_patient = cursor.fetchone()
        
        if existing_patient:
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
        
        # Create access token
        access_token = create_access_token(patient_id, request.email)
        
        # Log audit
        cursor.execute(
            "INSERT INTO AUDIT_LOG (patient_id, action, description, ip_address) VALUES (%s, %s, %s, %s)",
            (patient_id, "SIGNUP", "Patient account created", db_request.client.host)
        )
        conn.commit()
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            patient_id=patient_id,
            email=request.email,
            first_name=request.first_name,
            last_name=request.last_name
        )
    
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Signup failed: {str(e)}")
    finally:
        cursor.close()
        conn.close()

@app.post("/api/auth/login", response_model=TokenResponse)
async def login(request: LoginRequest, db_request: Request):
    """Login patient"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        # Find patient by email
        cursor.execute(
            "SELECT patient_id, email, first_name, last_name, password_hash FROM PATIENT WHERE email = %s",
            (request.email,)
        )
        patient = cursor.fetchone()
        
        if not patient or not verify_password(request.password, patient['password_hash']):
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        # Update last login
        cursor.execute(
            "UPDATE PATIENT SET last_login = NOW() WHERE patient_id = %s",
            (patient['patient_id'],)
        )
        
        # Log audit
        cursor.execute(
            "INSERT INTO AUDIT_LOG (patient_id, action, description, ip_address) VALUES (%s, %s, %s, %s)",
            (patient['patient_id'], "LOGIN", "Patient logged in", db_request.client.host)
        )
        
        conn.commit()
        
        # Create access token
        access_token = create_access_token(patient['patient_id'], patient['email'])
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            patient_id=patient['patient_id'],
            email=patient['email'],
            first_name=patient['first_name'],
            last_name=patient['last_name']
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")
    finally:
        cursor.close()
        conn.close()

# ==================== Patient Routes ====================
@app.get("/api/patient/profile", response_model=PatientProfile)
async def get_profile(current_patient: dict = Depends(get_current_patient)):
    """Get current patient profile"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        cursor.execute(
            "SELECT patient_id, email, first_name, last_name, age, gender, phone, medical_facility, role, created_at, last_login FROM PATIENT WHERE patient_id = %s",
            (current_patient['patient_id'],)
        )
        patient = cursor.fetchone()
        
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        return PatientProfile(**patient)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch profile: {str(e)}")
    finally:
        cursor.close()
        conn.close()

@app.get("/api/patient/diagnosis-history", response_model=DiagnosisHistory)
async def get_diagnosis_history(current_patient: dict = Depends(get_current_patient)):
    """Get patient's diagnosis history"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        # Get diagnosis history using the view
        cursor.execute("""
            SELECT 
                upload_id,
                filename,
                DATE_FORMAT(upload_date, '%Y-%m-%d %H:%i:%s') AS upload_date,
                status,
                classification,
                adhd_probability,
                confidence,
                risk_level,
                pdf_path
            FROM PATIENT_DIAGNOSIS_HISTORY
            WHERE patient_id = %s
            LIMIT 20
        """, (current_patient['patient_id'],))
        
        diagnoses = cursor.fetchall()
        diagnosis_records = [DiagnosisRecord(**record) for record in diagnoses]
        
        return DiagnosisHistory(
            patient_id=current_patient['patient_id'],
            total_diagnoses=len(diagnosis_records),
            recent_diagnoses=diagnosis_records
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch diagnosis history: {str(e)}")
    finally:
        cursor.close()
        conn.close()

@app.post("/api/auth/logout")
async def logout(current_patient: dict = Depends(get_current_patient), db_request: Request = None):
    """Logout patient"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Log audit
        cursor.execute(
            "INSERT INTO AUDIT_LOG (patient_id, action, description, ip_address) VALUES (%s, %s, %s, %s)",
            (current_patient['patient_id'], "LOGOUT", "Patient logged out", db_request.client.host if db_request else "unknown")
        )
        conn.commit()
        
        return {"message": "Logged out successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Logout failed: {str(e)}")
    finally:
        cursor.close()
        conn.close()

# ==================== Health Check ====================
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
