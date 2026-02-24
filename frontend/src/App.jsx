import React, { useState, useEffect } from 'react'
import './index.css'
import Home from './pages/Home'
import About from './pages/About'
import Upload from './pages/Upload'
import Resources from './pages/Resources'
import Navigation from './components/Navigation'
import Login from './components/Login'
import Signup from './components/Signup'
import Dashboard from './components/Dashboard'

export default function App() {
  const [currentPage, setCurrentPage] = useState('home')
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [authLoading, setAuthLoading] = useState(true)

  // Check if user is authenticated on app load
  useEffect(() => {
    const token = localStorage.getItem('token')
    setIsAuthenticated(!!token)
    setAuthLoading(false)
  }, [])

  // Handle logout
  const handleLogout = () => {
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    setIsAuthenticated(false)
    setCurrentPage('home')
  }

  // Render page based on auth state
  const renderPage = () => {
    // Show login/signup pages regardless of auth state
    if (currentPage === 'login') {
      return <Login onAuthSuccess={() => setIsAuthenticated(true)} onNavigate={setCurrentPage} redirectTo="dashboard" />
    }
    
    if (currentPage === 'signup') {
      return <Signup onAuthSuccess={() => setIsAuthenticated(true)} onNavigate={setCurrentPage} />
    }

    // Show dashboard when authenticated
    if (currentPage === 'dashboard') {
      return isAuthenticated ? (
        <Dashboard onLogout={handleLogout} onNavigate={setCurrentPage} />
      ) : (
        <Login onAuthSuccess={() => {
          setIsAuthenticated(true)
          setCurrentPage('dashboard')
        }} onNavigate={setCurrentPage} />
      )
    }

    // For upload page, require authentication
    if (currentPage === 'upload') {
      return isAuthenticated ? (
        <Upload />
      ) : (
        <Login onAuthSuccess={() => {
          setIsAuthenticated(true)
          setCurrentPage('upload')
        }} onNavigate={setCurrentPage} redirectTo="upload" />
      )
    }

    // Public pages - home, about, resources
    switch (currentPage) {
      case 'home':
        return <Home onNavigate={setCurrentPage} />
      case 'about':
        return <About />
      case 'resources':
        return <Resources />
      default:
        return <Home onNavigate={setCurrentPage} />
    }
  }

  if (authLoading) {
    return <div className="min-h-screen flex items-center justify-center">Loading...</div>
  }

  return (
    <div className="min-h-screen bg-slate-50">
      <Navigation 
        currentPage={currentPage} 
        onNavigate={setCurrentPage}
        isAuthenticated={isAuthenticated}
        onLogout={handleLogout}
      />
      <main className="pt-16">
        {renderPage()}
      </main>
    </div>
  )
}
