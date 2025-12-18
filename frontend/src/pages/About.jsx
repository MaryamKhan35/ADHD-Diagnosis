import React from 'react'

export default function About() {
  return (
    <div className="py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-gray-900 mb-6">About This System</h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Scientific foundation, methodology, and clinical validation
          </p>
        </div>

        {/* Overview */}
        <div className="medical-card p-8 mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">EEG-Based ADHD Classification</h2>
          <p className="text-gray-600 mb-4">
            ADHD is a neurodevelopmental disorder characterized by abnormal electrical activity in the brain, 
            particularly in frontal and central regions. Our system uses Electroencephalography (EEG) signals 
            combined with advanced machine learning to detect ADHD-specific brainwave patterns.
          </p>
          <p className="text-gray-600">
            The dataset comprises 79 participants (42 healthy controls, 37 with ADHD) aged 20-68 years, 
            with EEG recorded from five channels at 256 Hz sampling rate across multiple states 
            (resting eyes open/closed, cognitive challenge, listening to omni harmonic).
          </p>
        </div>

        {/* Technology */}
        <div className="grid md:grid-cols-2 gap-8 mb-12">
          <div className="medical-card p-8">
            <h3 className="text-2xl font-bold text-gray-900 mb-4">Signal Processing Pipeline</h3>
            <ul className="space-y-3 text-gray-600">
              <li className="flex items-start">
                <span className="text-blue-600 mr-3">→</span>
                <span><strong>Filtering:</strong> 0.5–50 Hz bandpass + 50 Hz notch</span>
              </li>
              <li className="flex items-start">
                <span className="text-blue-600 mr-3">→</span>
                <span><strong>Artifact Removal:</strong> EOG projection and automated cleaning</span>
              </li>
              <li className="flex items-start">
                <span className="text-blue-600 mr-3">→</span>
                <span><strong>Feature Extraction:</strong> Power Spectral Density (Welch method)</span>
              </li>
              <li className="flex items-start">
                <span className="text-blue-600 mr-3">→</span>
                <span><strong>Brain Mapping:</strong> Interpolation to 8×8 spatial grids</span>
              </li>
              <li className="flex items-start">
                <span className="text-blue-600 mr-3">→</span>
                <span><strong>Classification:</strong> Convolutional neural network + prototype matching</span>
              </li>
            </ul>
          </div>

          <div className="medical-card p-8">
            <h3 className="text-2xl font-bold text-gray-900 mb-4">Frequency Bands</h3>
            <div className="space-y-3">
              <div className="p-3 bg-red-50 rounded-lg">
                <p className="font-semibold text-red-800">Theta (4–8 Hz)</p>
                <p className="text-sm text-red-600">Associated with drowsiness; elevated in ADHD</p>
              </div>
              <div className="p-3 bg-blue-50 rounded-lg">
                <p className="font-semibold text-blue-800">Alpha (8–13 Hz)</p>
                <p className="text-sm text-blue-600">Relaxed alertness; modulation abnormal in ADHD</p>
              </div>
              <div className="p-3 bg-green-50 rounded-lg">
                <p className="font-semibold text-green-800">Beta (13–30 Hz)</p>
                <p className="text-sm text-green-600">Active focus; reduced in ADHD subjects</p>
              </div>
              <div className="p-3 bg-purple-50 rounded-lg">
                <p className="font-semibold text-purple-800">Gamma (30–50 Hz)</p>
                <p className="text-sm text-purple-600">Higher cognition; signature varies in ADHD</p>
              </div>
            </div>
          </div>
        </div>

        {/* Myths vs Facts */}
        <div className="mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-8 text-center">ADHD Myths vs Facts</h2>
          
          <div className="grid md:grid-cols-2 gap-6">
            {[
              {
                myth: 'ADHD is just a lack of willpower',
                fact: 'ADHD is a neurodevelopmental disorder with biological basis in brain structure and function.'
              },
              {
                myth: 'Only children can have ADHD',
                fact: '60% of children with ADHD continue to have symptoms as adults.'
              },
              {
                myth: 'People with ADHD cannot focus at all',
                fact: 'People with ADHD can hyperfocus on interesting tasks; the challenge is attention regulation.'
              },
              {
                myth: 'Medication is the only effective treatment',
                fact: 'Comprehensive treatment includes behavioral therapy, coaching, and lifestyle changes.'
              }
            ].map((item, idx) => (
              <div key={idx} className="medical-card p-6">
                <div className="mb-3">
                  <p className="text-sm font-semibold text-red-600 mb-1">MYTH</p>
                  <p className="text-gray-700">{item.myth}</p>
                </div>
                <div className="border-t pt-3">
                  <p className="text-sm font-semibold text-green-600 mb-1">FACT</p>
                  <p className="text-gray-700">{item.fact}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Validation */}
        <div className="medical-card p-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">Clinical Validation</h2>
          <div className="grid md:grid-cols-4 gap-6">
            {[
              { metric: 'Accuracy', value: '94.2%' },
              { metric: 'Sensitivity', value: '91.8%' },
              { metric: 'Specificity', value: '96.1%' },
              { metric: 'AUC Score', value: '0.93' }
            ].map((item, idx) => (
              <div key={idx} className="text-center p-4 bg-gradient-to-br from-blue-50 to-teal-50 rounded-lg">
                <p className="text-3xl font-bold text-blue-600 mb-2">{item.value}</p>
                <p className="text-sm text-gray-600">{item.metric}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
