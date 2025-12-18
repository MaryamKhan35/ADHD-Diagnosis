import React, { useState } from 'react'
import './index.css'
import Home from './pages/Home'
import About from './pages/About'
import Upload from './pages/Upload'
import Resources from './pages/Resources'
import Navigation from './components/Navigation'

export default function App() {
  const [currentPage, setCurrentPage] = useState('home')

  const renderPage = () => {
    switch (currentPage) {
      case 'home': return <Home onNavigate={setCurrentPage} />
      case 'about': return <About />
      case 'upload': return <Upload />
      case 'resources': return <Resources />
      default: return <Home onNavigate={setCurrentPage} />
    }
  }

  return (
    <div className="min-h-screen bg-slate-50">
      <Navigation currentPage={currentPage} onNavigate={setCurrentPage} />
      <main className="pt-16">
        {renderPage()}
      </main>
    </div>
  )
}
