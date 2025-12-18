import React from 'react'

export default function Home({ onNavigate }) {
  return (
    <div className="bg-gradient-to-br from-blue-50 to-teal-50 min-h-screen">
      {/* Hero Section */}
      <section className="pt-24 pb-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-12 fade-in">
            <h1 className="text-5xl md:text-6xl font-bold text-gray-900 mb-6">
              EEG-Based ADHD Classification System
            </h1>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto mb-8">
              Advanced neuroimaging technology powered by machine learning to provide 
              accurate, non-invasive ADHD diagnosis through EEG signal analysis.
            </p>
            <button 
              className="btn-primary text-lg px-8 py-4"
              onClick={() => onNavigate('upload')}
            >
              Get Started
            </button>
          </div>

          {/* Features Grid */}
          <div className="grid md:grid-cols-3 gap-8 mt-12">
            <div className="medical-card p-8 fade-in">
              <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
                <span className="text-blue-600 text-xl">🧠</span>
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-3">Advanced Technology</h3>
              <p className="text-gray-600">
                Utilizes cutting-edge EEG signal processing and deep learning neural networks 
                to detect ADHD-specific brainwave patterns.
              </p>
            </div>

            <div className="medical-card p-8 fade-in">
              <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mb-4">
                <span className="text-green-600 text-xl">✓</span>
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-3">Clinically Validated</h3>
              <p className="text-gray-600">
                Built on peer-reviewed research and validated across diverse patient 
                populations with high accuracy and sensitivity.
              </p>
            </div>

            <div className="medical-card p-8 fade-in">
              <div className="w-12 h-12 bg-amber-100 rounded-lg flex items-center justify-center mb-4">
                <span className="text-amber-600 text-xl">⚡</span>
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-3">Quick Results</h3>
              <p className="text-gray-600">
                Get diagnostic results in seconds. Designed for healthcare professionals 
                to streamline the assessment workflow.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-16 px-4 sm:px-6 lg:px-8 bg-white">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-4xl font-bold text-center text-gray-900 mb-12">How It Works</h2>
          
          <div className="grid md:grid-cols-4 gap-6">
            {[
              { step: 1, title: 'Upload EEG Data', desc: 'Submit patient EEG recordings in standard formats' },
              { step: 2, title: 'Signal Processing', desc: 'Advanced filtering and artifact removal' },
              { step: 3, title: 'Brain Mapping', desc: 'PSD analysis converted to brain topography maps' },
              { step: 4, title: 'Classification', desc: 'Neural network predicts ADHD risk probability' }
            ].map(item => (
              <div key={item.step} className="relative">
                <div className="medical-card p-6 text-center">
                  <div className="w-12 h-12 bg-blue-600 text-white rounded-full flex items-center justify-center mx-auto mb-4 text-xl font-bold">
                    {item.step}
                  </div>
                  <h3 className="font-bold text-gray-900 mb-2">{item.title}</h3>
                  <p className="text-sm text-gray-600">{item.desc}</p>
                </div>
                {item.step < 4 && (
                  <div className="hidden md:block absolute right-0 top-1/2 transform translate-x-1/2 -translate-y-1/2">
                    <span className="text-2xl text-blue-600">→</span>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-16 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl font-bold text-gray-900 mb-6">Ready to Diagnose?</h2>
          <p className="text-lg text-gray-600 mb-8">
            Upload EEG data and get instant ADHD classification results powered by 
            advanced machine learning.
          </p>
          <button 
            className="btn-primary text-lg px-8 py-4"
            onClick={() => onNavigate('upload')}
          >
            Start Assessment
          </button>
        </div>
      </section>
    </div>
  )
}
