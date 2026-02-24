import React, { useState } from 'react'
import axios from 'axios'

const API_URL = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '')

export default function Upload() {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [previewMap, setPreviewMap] = useState(null)
  const [error, setError] = useState(null)

  const handleFileChange = (e) => {
    setFile(e.target.files[0])
    setError(null)
  }

  const handlePreprocess = async () => {
    if (!file) {
      setError('Please select a file')
      return
    }
    
    setLoading(true)
    setError(null)
    const formData = new FormData()
    formData.append('file', file)
    
    try {
      const response = await axios.post(`${API_URL}/preprocess`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 120000, // 120 second timeout
      })
      
      if (response.data.error) {
        setError(`Preprocessing failed: ${response.data.error}`)
        if (response.data.traceback) {
          console.error('Backend traceback:', response.data.traceback)
        }
      } else {
        setPreviewMap(response.data.map_png_base64)
        setError(null)
      }
    } catch (err) {
      const errorMsg = err.response?.data?.error || err.message || 'Unknown error'
      setError(`Preprocessing failed: ${errorMsg}`)
      if (err.response?.data?.traceback) {
        console.error('Backend traceback:', err.response.data.traceback)
      }
      console.error('Full error:', err)
    } finally {
      setLoading(false)
    }
  }

  const handlePredict = async () => {
    if (!file) {
      setError('Please select a file')
      return
    }
    
    setLoading(true)
    setError(null)
    const formData = new FormData()
    formData.append('file', file)
    
    try {
      const token = localStorage.getItem('token')
      const response = await axios.post(`${API_URL}/api/patient/upload-and-predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          'Authorization': `Bearer ${token}`
        },
        timeout: 120000, // 120 second timeout
      })
      
      if (response.data.error) {
        setError(`Prediction failed: ${response.data.error}`)
        if (response.data.traceback) {
          console.error('Backend traceback:', response.data.traceback)
        }
      } else {
        setResult({
          probability: (response.data.probability_adhd * 100).toFixed(1),
          classification: response.data.classification,
          uploadId: response.data.upload_id,
          map: response.data.map_png_base64,
          message: response.data.message
        })
        setPreviewMap(response.data.map_png_base64)
        setError(null)
      }
    } catch (err) {
      const errorMsg = err.response?.data?.error || err.message || 'Unknown error'
      setError(`Prediction failed: ${errorMsg}`)
      if (err.response?.data?.traceback) {
        console.error('Backend traceback:', err.response.data.traceback)
      }
      console.error('Full error:', err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-gray-900 mb-8">EEG Upload & Classification</h1>
        
        <div className="grid md:grid-cols-2 gap-8">
          {/* Upload Panel */}
          <div className="medical-card p-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Upload EEG File</h2>
            
            <div className="border-2 border-dashed border-blue-300 rounded-lg p-8 mb-6 text-center hover:border-blue-500 transition">
              <input
                type="file"
                accept=".mat,.csv"
                onChange={handleFileChange}
                className="hidden"
                id="file-input"
              />
              <label htmlFor="file-input" className="cursor-pointer">
                <div className="text-4xl mb-2">📁</div>
                <p className="text-gray-600 font-medium">
                  {file ? file.name : 'Drop MAT file here or click to browse'}
                </p>
              </label>
            </div>

            {error && (
              <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-6">
                {error}
              </div>
            )}

            <div className="space-y-3">
              <button
                onClick={handlePreprocess}
                disabled={!file || loading}
                className="w-full bg-blue-600 text-white py-2 rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-400 transition"
              >
                {loading ? 'Processing...' : 'Preview PSD Map'}
              </button>
              
              <button
                onClick={handlePredict}
                disabled={!file || loading}
                className="w-full bg-green-600 text-white py-2 rounded-lg font-medium hover:bg-green-700 disabled:bg-gray-400 transition"
              >
                {loading ? 'Predicting...' : 'Get ADHD Prediction'}
              </button>
            </div>
          </div>

          {/* Results Panel */}
          <div>
            {previewMap && (
              <div className="medical-card p-8 mb-6">
                <h3 className="text-lg font-bold text-gray-900 mb-4">PSD Brain Map</h3>
                <img
                  src={`data:image/png;base64,${previewMap}`}
                  alt="PSD Brain Map"
                  className="w-full rounded-lg"
                />
              </div>
            )}

            {result && (
              <div className="medical-card p-8 bg-gradient-to-br from-green-50 to-blue-50">
                <h3 className="text-2xl font-bold text-gray-900 mb-4">Classification Result</h3>
                
                {result.message && (
                  <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded mb-4">
                    ✓ {result.message}
                  </div>
                )}
                
                <div className="grid grid-cols-2 gap-4 mb-6">
                  <div>
                    <p className="text-gray-600 mb-2">Classification</p>
                    <div className={`text-2xl font-bold ${result.classification === 'ADHD' ? 'text-red-600' : 'text-green-600'}`}>
                      {result.classification}
                    </div>
                  </div>
                  <div>
                    <p className="text-gray-600 mb-2">Upload ID</p>
                    <div className="text-lg font-mono text-blue-600">
                      #{result.uploadId}
                    </div>
                  </div>
                </div>
                
                <div className="mb-6">
                  <p className="text-gray-600 mb-2">ADHD Risk Probability</p>
                  <div className="text-5xl font-bold text-blue-600 mb-2">
                    {result.probability}%
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                    <div
                      className="bg-gradient-to-r from-green-500 to-red-500 h-full transition-all"
                      style={{ width: `${result.probability}%` }}
                    />
                  </div>
                </div>

                <div className="bg-white p-4 rounded-lg border border-gray-200">
                  <p className="text-sm text-gray-600">
                    {result.probability < 30 ? (
                      <span className="text-green-600 font-medium">✓ Low ADHD risk. Further clinical evaluation recommended for confirmation.</span>
                    ) : result.probability < 70 ? (
                      <span className="text-amber-600 font-medium">⚠ Moderate ADHD risk. Clinical consultation advised.</span>
                    ) : (
                      <span className="text-red-600 font-medium">! High ADHD risk. Clinical consultation strongly recommended.</span>
                    )}
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
