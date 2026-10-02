import { useState } from 'react'
import { BrowserRouter, Routes, Route, Navigate, useNavigate } from 'react-router-dom'
import Login from './components/Login'
import UserDashboard from './pages/UserDashboard'
import AdminDashboard from './pages/AdminDashboard'

import AnythingLLMApp from './components/AnythingLLMApp'

function AuthBoundary() {
  const [token, setToken] = useState<string | null>(localStorage.getItem('token'))
  const navigate = useNavigate()

  const handleLogin = (jwt: string) => {
    setToken(jwt)
    localStorage.setItem('token', jwt)
    navigate('/')
  }

  const handleLogout = () => {
    setToken(null)
    localStorage.removeItem('token')
    navigate('/login', { replace: true })
  }

  return (
    <Routes>
      <Route path="/login" element={!token ? <Login onLogin={handleLogin} /> : <Navigate to="/" replace />} />
      
      {/* Protected Routes - Render Unified Department RAG Workspace App */}
      <Route path="/admin/*" element={token ? <AnythingLLMApp token={token} onLogout={handleLogout} /> : <Navigate to="/login" replace />} />
      <Route path="/app/*" element={token ? <AnythingLLMApp token={token} onLogout={handleLogout} /> : <Navigate to="/login" replace />} />
      <Route path="/workspace/*" element={token ? <AnythingLLMApp token={token} onLogout={handleLogout} /> : <Navigate to="/login" replace />} />
      
      {/* Root redirection */}
      <Route path="/" element={token ? <Navigate to="/app" replace /> : <Navigate to="/login" replace />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthBoundary />
    </BrowserRouter>
  )
}
