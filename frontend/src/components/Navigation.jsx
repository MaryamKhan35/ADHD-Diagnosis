import React from 'react'

export default function Navigation({ currentPage, onNavigate, isAuthenticated, onLogout }) {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 glass-effect border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center">
            <button
              onClick={() => onNavigate('home')}
              className="text-xl font-bold text-gray-900 hover:text-blue-600 transition"
            >
              EEG-ADHD Classifier
            </button>
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
              
              {isAuthenticated && (
                <>
                  <button
                    onClick={() => onNavigate('dashboard')}
                    className={`px-3 py-2 text-sm font-medium transition ${
                      currentPage === 'dashboard'
                        ? 'text-blue-600'
                        : 'text-gray-700 hover:text-blue-600'
                    }`}
                  >
                    Dashboard
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
                </>
              )}
              
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

          {/* Auth Buttons */}
          <div className="flex items-center space-x-4">
            {isAuthenticated ? (
              <button
                onClick={onLogout}
                className="px-4 py-2 text-sm font-medium text-white bg-red-600 hover:bg-red-700 rounded-md transition"
              >
                Logout
              </button>
            ) : (
              <>
                <button
                  onClick={() => onNavigate('login')}
                  className={`px-4 py-2 text-sm font-medium transition ${
                    currentPage === 'login'
                      ? 'text-blue-600'
                      : 'text-gray-700 hover:text-blue-600'
                  }`}
                >
                  Login
                </button>
                <button
                  onClick={() => onNavigate('signup')}
                  className={`px-4 py-2 text-sm font-medium text-white ${
                    currentPage === 'signup'
                      ? 'bg-blue-700'
                      : 'bg-blue-600 hover:bg-blue-700'
                  } rounded-md transition`}
                >
                  Sign Up
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    </nav>
  )
}
