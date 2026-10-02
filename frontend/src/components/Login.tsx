import { useState } from 'react'
import { login } from '../api/auth'

export default function Login({ onLogin }: { onLogin: (token: string) => void }) {
  const [username, setUsername] = useState('admin@local.dev')
  const [password, setPassword] = useState('Admin@12345')
  const [loginError, setLoginError] = useState('')
  const [isLoggingIn, setIsLoggingIn] = useState(false)

  async function handleLogin(e: React.FormEvent) {
    e.preventDefault()
    setLoginError('')
    setIsLoggingIn(true)
    try {
      const jwt = await login(username, password)
      onLogin(jwt)
    } catch (err: any) {
      setLoginError(err.message || 'Login failed')
    } finally {
      setIsLoggingIn(false)
    }
  }

  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '100dvh', background: 'var(--color-ground)', fontFamily: 'var(--font-sans)' }}>
      <form onSubmit={handleLogin} style={{ background: 'var(--color-surface)', padding: '32px', borderRadius: '16px', border: '1px solid var(--color-border)', width: '100%', maxWidth: '360px', boxShadow: '0 12px 34px rgba(21, 29, 50, 0.05)' }}>
        <h2 style={{ margin: '0 0 24px', color: 'var(--color-ink)', fontSize: '20px' }}>Log in to Workspace</h2>
        <div style={{ marginBottom: '16px' }}>
          <label style={{ display: 'block', fontSize: '13px', marginBottom: '6px', color: 'var(--color-ink-muted)' }}>Email</label>
          <input type="text" value={username} onChange={e => setUsername(e.target.value)} style={{ width: '100%', boxSizing: 'border-box', padding: '10px 12px', borderRadius: '8px', border: '1px solid var(--color-border)', background: 'var(--color-ground)', color: 'var(--color-ink)' }} />
        </div>
        <div style={{ marginBottom: '24px' }}>
          <label style={{ display: 'block', fontSize: '13px', marginBottom: '6px', color: 'var(--color-ink-muted)' }}>Password</label>
          <input type="password" value={password} onChange={e => setPassword(e.target.value)} style={{ width: '100%', boxSizing: 'border-box', padding: '10px 12px', borderRadius: '8px', border: '1px solid var(--color-border)', background: 'var(--color-ground)', color: 'var(--color-ink)' }} />
        </div>
        {loginError && <div style={{ color: '#d93025', fontSize: '13px', marginBottom: '16px' }}>{loginError}</div>}
        <button type="submit" disabled={isLoggingIn} style={{ width: '100%', padding: '10px', borderRadius: '8px', border: 'none', background: 'var(--color-teal)', color: 'white', fontWeight: 500, cursor: isLoggingIn ? 'default' : 'pointer', transition: 'background 0.2s' }}>
          {isLoggingIn ? 'Logging in...' : 'Log in'}
        </button>
      </form>
    </div>
  )
}
