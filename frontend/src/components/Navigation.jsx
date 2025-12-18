import React from 'react'

export default function Navigation({ currentPage, onNavigate }) {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 glass-effect">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center">
            <h1 className="text-xl font-bold text-gray-900">EEG-ADHD Classifier</h1>
          </div>
          <div className="hidden md:block">
            <div className="ml-10 flex items-baseline space-x-8">
              <button
                onClick={() => onNavigate('home')}
                className={`px-3 py-2 text-sm font-medium transition ${
                  currentPage === 'home'
                    ? 'text-blue-600'
                    : 'text-gray-700 hover:text-blue-600'
                }`}
              >
                Home
              </button>
              <button
                onClick={() => onNavigate('about')}
                className={`px-3 py-2 text-sm font-medium transition ${
                  currentPage === 'about'
                    ? 'text-blue-600'
                    : 'text-gray-700 hover:text-blue-600'
                }`}
              >
                About
              </button>
              <button
                onClick={() => onNavigate('upload')}
                className={`px-3 py-2 text-sm font-medium transition ${
                  currentPage === 'upload'
                    ? 'text-blue-600'
                    : 'text-gray-700 hover:text-blue-600'
                }`}
              >
                Upload & Predict
              </button>
              <button
                onClick={() => onNavigate('resources')}
                className={`px-3 py-2 text-sm font-medium transition ${
                  currentPage === 'resources'
                    ? 'text-blue-600'
                    : 'text-gray-700 hover:text-blue-600'
                }`}
              >
                Resources
              </button>
            </div>
          </div>
        </div>
      </div>
    </nav>
  )
}
