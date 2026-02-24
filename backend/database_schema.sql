-- EEG ADHD Detection System Database Schema
-- MySQL Database Setup

CREATE DATABASE IF NOT EXISTS eeg_adhd_detection;
USE eeg_adhd_detection;

-- PATIENT Table: Store user credentials and profile information
CREATE TABLE PATIENT (
    patient_id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    age INT,
    gender ENUM('Male', 'Female', 'Other'),
    phone VARCHAR(20),
    medical_facility VARCHAR(255),
    role ENUM('Patient', 'Clinician', 'Admin') DEFAULT 'Patient',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    last_login TIMESTAMP NULL,
    INDEX idx_email (email),
    INDEX idx_created_at (created_at)
);

-- EEG_UPLOAD Table: Store EEG file uploads and processing status
CREATE TABLE EEG_UPLOAD (
    upload_id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT NOT NULL,
    filename VARCHAR(255) NOT NULL,
    file_size BIGINT,
    file_path VARCHAR(500),
    status ENUM('pending', 'processing', 'completed', 'failed') DEFAULT 'pending',
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    started_at TIMESTAMP NULL,
    completed_at TIMESTAMP NULL,
    FOREIGN KEY (patient_id) REFERENCES PATIENT(patient_id) ON DELETE CASCADE,
    INDEX idx_patient_id (patient_id),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
);

-- EEG_DATA Table: Store raw and processed EEG data
CREATE TABLE EEG_DATA (
    data_id INT AUTO_INCREMENT PRIMARY KEY,
    upload_id INT NOT NULL,
    raw_data_path VARCHAR(500),
    processed_data_path VARCHAR(500),
    n_samples INT,
    duration_seconds INT,
    sampling_rate INT DEFAULT 256,
    channels INT DEFAULT 2,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (upload_id) REFERENCES EEG_UPLOAD(upload_id) ON DELETE CASCADE
);

-- PREPROCESSING_STAGE Table: Track each preprocessing stage
CREATE TABLE PREPROCESSING_STAGE (
    stage_id INT AUTO_INCREMENT PRIMARY KEY,
    upload_id INT NOT NULL,
    stage_number INT,
    stage_name VARCHAR(100),
    status ENUM('pending', 'processing', 'completed', 'failed') DEFAULT 'pending',
    start_time TIMESTAMP NULL,
    end_time TIMESTAMP NULL,
    processing_time_seconds INT,
    error_message TEXT,
    FOREIGN KEY (upload_id) REFERENCES EEG_UPLOAD(upload_id) ON DELETE CASCADE,
    INDEX idx_upload_id (upload_id)
);

-- EEG_MODEL Table: Store model versions and metadata
CREATE TABLE EEG_MODEL (
    model_id INT AUTO_INCREMENT PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    model_path VARCHAR(500),
    accuracy DECIMAL(5, 2),
    sensitivity DECIMAL(5, 2),
    specificity DECIMAL(5, 2),
    auc_score DECIMAL(5, 2),
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY unique_model_version (model_name, model_version),
    INDEX idx_is_active (is_active)
);

-- PREDICTION_RESULT Table: Store final classification results
CREATE TABLE PREDICTION_RESULT (
    result_id INT AUTO_INCREMENT PRIMARY KEY,
    upload_id INT NOT NULL,
    model_id INT NOT NULL,
    classification ENUM('ADHD', 'Control') NOT NULL,
    adhd_probability DECIMAL(4, 3),
    confidence DECIMAL(4, 3),
    adhd_windows INT,
    control_windows INT,
    total_windows INT DEFAULT 563,
    mean_probability DECIMAL(4, 3),
    std_dev_probability DECIMAL(4, 3),
    min_probability DECIMAL(4, 3),
    max_probability DECIMAL(4, 3),
    processing_time_seconds INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (upload_id) REFERENCES EEG_UPLOAD(upload_id) ON DELETE CASCADE,
    FOREIGN KEY (model_id) REFERENCES EEG_MODEL(model_id),
    INDEX idx_upload_id (upload_id),
    INDEX idx_classification (classification),
    INDEX idx_created_at (created_at)
);

-- MEDICAL_REPORT Table: Store generated clinical reports
CREATE TABLE MEDICAL_REPORT (
    report_id INT AUTO_INCREMENT PRIMARY KEY,
    result_id INT NOT NULL,
    patient_id INT NOT NULL,
    summary TEXT,
    findings TEXT,
    risk_level ENUM('Low', 'Moderate', 'High') DEFAULT 'Moderate',
    clinical_notes TEXT,
    pdf_path VARCHAR(500),
    report_generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    clinician_reviewed BOOLEAN DEFAULT FALSE,
    clinician_notes TEXT,
    reviewed_at TIMESTAMP NULL,
    FOREIGN KEY (result_id) REFERENCES PREDICTION_RESULT(result_id) ON DELETE CASCADE,
    FOREIGN KEY (patient_id) REFERENCES PATIENT(patient_id) ON DELETE CASCADE,
    INDEX idx_patient_id (patient_id),
    INDEX idx_report_generated_at (report_generated_at)
);

-- AUDIT_LOG Table: Track system activities for compliance
CREATE TABLE AUDIT_LOG (
    log_id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT,
    action VARCHAR(100),
    description TEXT,
    ip_address VARCHAR(45),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES PATIENT(patient_id) ON DELETE SET NULL,
    INDEX idx_patient_id (patient_id),
    INDEX idx_timestamp (timestamp)
);

-- PATIENT_SESSION Table: Track active sessions
CREATE TABLE PATIENT_SESSION (
    session_id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT NOT NULL,
    token VARCHAR(500) NOT NULL UNIQUE,
    ip_address VARCHAR(45),
    user_agent VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    FOREIGN KEY (patient_id) REFERENCES PATIENT(patient_id) ON DELETE CASCADE,
    INDEX idx_patient_id (patient_id),
    INDEX idx_token (token),
    INDEX idx_expires_at (expires_at)
);

-- Insert default EEG model
INSERT INTO EEG_MODEL (model_name, model_version, accuracy, sensitivity, specificity, auc_score, is_active)
VALUES ('EncoderCNN', '1.0.0', 94.89, 95.67, 93.45, 98.22, TRUE);

-- Create views for common queries
CREATE VIEW PATIENT_DIAGNOSIS_HISTORY AS
SELECT 
    p.patient_id,
    p.email,
    p.first_name,
    p.last_name,
    eu.upload_id,
    eu.filename,
    eu.created_at AS upload_date,
    eu.status,
    pr.classification,
    pr.adhd_probability,
    pr.confidence,
    mr.risk_level,
    mr.pdf_path
FROM PATIENT p
LEFT JOIN EEG_UPLOAD eu ON p.patient_id = eu.patient_id
LEFT JOIN PREDICTION_RESULT pr ON eu.upload_id = pr.upload_id
LEFT JOIN MEDICAL_REPORT mr ON pr.result_id = mr.result_id
ORDER BY eu.created_at DESC;
