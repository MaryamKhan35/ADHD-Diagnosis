import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './Auth.css';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const Login = ({ onAuthSuccess, redirectTo = 'dashboard', onNavigate }) => {
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  });

  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);
  const [serverError, setServerError] = useState('');
  const [rememberMe, setRememberMe] = useState(false);

  // Load remembered email on mount
  useEffect(() => {
    const rememberedEmail = localStorage.getItem('remembered_email');
    if (rememberedEmail) {
      setFormData(prev => ({
        ...prev,
        email: rememberedEmail
      }));
      setRememberMe(true);
    }
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    // Clear error for this field
    if (errors[name]) {
      setErrors(prev => ({
        ...prev,
        [name]: ''
      }));
    }
  };

  const validateForm = () => {
    const newErrors = {};

    if (!formData.email) newErrors.email = 'Email is required';
    else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) newErrors.email = 'Invalid email format';

    if (!formData.password) newErrors.password = 'Password is required';

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setServerError('');

    if (!validateForm()) return;

    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/api/auth/login`, {
        email: formData.email,
        password: formData.password
      });

      // Save token to localStorage
      localStorage.setItem('token', response.data.access_token);
      localStorage.setItem('patient_id', response.data.patient_id);
      localStorage.setItem('user', JSON.stringify({
        patient_id: response.data.patient_id,
        email: response.data.email,
        first_name: response.data.first_name,
        last_name: response.data.last_name
      }));

      // Remember me functionality
      if (rememberMe) {
        localStorage.setItem('remember_me', 'true');
        localStorage.setItem('remembered_email', formData.email);
      } else {
        localStorage.removeItem('remembered_email');
      }

      // Call the callback to update parent component auth state
      if (onAuthSuccess) {
        onAuthSuccess();
      }

      // Navigate to the appropriate page
      if (onNavigate) {
        onNavigate(redirectTo);
      }
    } catch (error) {
      setServerError(error.response?.data?.detail || 'Login failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleSignupClick = () => {
    if (onNavigate) {
      onNavigate('signup');
    }
  };

  return (
    <div className="auth-container login-container">
      <div className="auth-card">
        <div className="auth-header">
          <div className="logo-icon">
            <svg viewBox="0 0 200 200" width="60" height="60">
              <circle cx="100" cy="100" r="90" fill="none" stroke="#3498db" strokeWidth="2"/>
              <path d="M 50 100 Q 100 70 150 100" fill="none" stroke="#3498db" strokeWidth="3"/>
              <circle cx="70" cy="90" r="5" fill="#3498db"/>
              <circle cx="130" cy="90" r="5" fill="#3498db"/>
            </svg>
          </div>
          <h2>Welcome Back</h2>
          <p>EEG ADHD Detection System</p>
        </div>

        {serverError && (
          <div className="alert alert-error">
            {serverError}
          </div>
        )}

        <form onSubmit={handleSubmit} className="auth-form">
          <div className="form-group">
            <label htmlFor="email">Email Address</label>
            <input
              id="email"
              type="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              placeholder="your@email.com"
              className={errors.email ? 'input-error' : ''}
              disabled={loading}
            />
            {errors.email && <span className="error-text">{errors.email}</span>}
          </div>

          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input
              id="password"
              type="password"
              name="password"
              value={formData.password}
              onChange={handleChange}
              placeholder="Enter your password"
              className={errors.password ? 'input-error' : ''}
              disabled={loading}
            />
            {errors.password && <span className="error-text">{errors.password}</span>}
          </div>

          <div className="form-options">
            <div className="checkbox-group">
              <input
                type="checkbox"
                id="rememberMe"
                checked={rememberMe}
                onChange={(e) => setRememberMe(e.target.checked)}
              />
              <label htmlFor="rememberMe">Remember me</label>
            </div>
            <button type="button" className="forgot-password" disabled>
              Forgot password?
            </button>
          </div>

          <button type="submit" className="auth-button" disabled={loading}>
            {loading ? 'Logging in...' : 'Login'}
          </button>
        </form>

        <div className="auth-divider">
          <span>or</span>
        </div>

        <p className="auth-footer">
          Don't have an account? <button type="button" onClick={handleSignupClick} className="link-button">Create one here</button>
        </p>
      </div>

      <div className="auth-info">
        <h3>About EEG ADHD Detection</h3>
        <ul>
          <li>🧠 Advanced AI-powered diagnosis</li>
          <li>⚡ Results in 12-15 minutes</li>
          <li>🔒 Secure & HIPAA compliant</li>
          <li>📊 Dashboard with diagnosis history</li>
        </ul>
      </div>
    </div>
  );
};

export default Login;
