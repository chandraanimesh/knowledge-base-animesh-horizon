import { useState } from 'react'
import { login, register } from '../api/auth'

export default function Login({ onLogin }: { onLogin: (token: string) => void }) {
  const [tab, setTab] = useState<'login' | 'register'>('login')
  const [username, setUsername] = useState('admin@local.dev')
  const [password, setPassword] = useState('Admin@12345')
  const [regEmail, setRegEmail] = useState('')
  const [regPassword, setRegPassword] = useState('')
  const [regDept, setRegDept] = useState('')
  const [regReason, setRegReason] = useState('')
  const [errorMsg, setErrorMsg] = useState('')
  const [successMsg, setSuccessMsg] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)

  async function handleLogin(e: React.FormEvent) {
    e.preventDefault()
    setErrorMsg('')
    setSuccessMsg('')
    setIsSubmitting(true)
    try {
      const jwt = await login(username, password)
      onLogin(jwt)
    } catch (err: any) {
      setErrorMsg(err.message || 'Login failed')
    } finally {
      setIsSubmitting(false)
    }
  }

  async function handleRegister(e: React.FormEvent) {
    e.preventDefault()
    setErrorMsg('')
    setSuccessMsg('')
    setIsSubmitting(true)
    try {
      const res = await register(regEmail, regPassword, regDept, regReason)
      setSuccessMsg(res.message || 'Registration request submitted. Pending administrator approval.')
      setRegEmail('')
      setRegPassword('')
      setRegDept('')
      setRegReason('')
    } catch (err: any) {
      setErrorMsg(err.message || 'Registration failed')
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '100dvh', background: 'var(--color-ground)', fontFamily: 'var(--font-sans)' }}>
      <div style={{ background: 'var(--color-surface)', padding: '32px', borderRadius: '16px', border: '1px solid var(--color-border)', width: '100%', maxWidth: '380px', boxShadow: '0 12px 34px rgba(21, 29, 50, 0.05)' }}>
        
        {/* Tab Headers */}
        <div style={{ display: 'flex', borderBottom: '1px solid var(--color-border)', marginBottom: '24px' }}>
          <button
            type="button"
            onClick={() => { setTab('login'); setErrorMsg(''); setSuccessMsg('') }}
            style={{
              flex: 1,
              padding: '10px 0',
              background: 'none',
              border: 'none',
              borderBottom: tab === 'login' ? '2px solid var(--color-teal)' : '2px solid transparent',
              color: tab === 'login' ? 'var(--color-teal)' : 'var(--color-ink-muted)',
              fontWeight: tab === 'login' ? 600 : 400,
              cursor: 'pointer',
            }}
          >
            Log In
          </button>
          <button
            type="button"
            onClick={() => { setTab('register'); setErrorMsg(''); setSuccessMsg('') }}
            style={{
              flex: 1,
              padding: '10px 0',
              background: 'none',
              border: 'none',
              borderBottom: tab === 'register' ? '2px solid var(--color-teal)' : '2px solid transparent',
              color: tab === 'register' ? 'var(--color-teal)' : 'var(--color-ink-muted)',
              fontWeight: tab === 'register' ? 600 : 400,
              cursor: 'pointer',
            }}
          >
            Register
          </button>
        </div>

        {tab === 'login' ? (
          <form onSubmit={handleLogin}>
            <h2 style={{ margin: '0 0 20px', color: 'var(--color-ink)', fontSize: '18px' }}>Log in to Workspace</h2>
            <div style={{ marginBottom: '16px' }}>
              <label style={{ display: 'block', fontSize: '13px', marginBottom: '6px', color: 'var(--color-ink-muted)' }}>Email</label>
              <input type="text" value={username} onChange={e => setUsername(e.target.value)} required style={{ width: '100%', boxSizing: 'border-box', padding: '10px 12px', borderRadius: '8px', border: '1px solid var(--color-border)', background: 'var(--color-ground)', color: 'var(--color-ink)' }} />
            </div>
            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', fontSize: '13px', marginBottom: '6px', color: 'var(--color-ink-muted)' }}>Password</label>
              <input type="password" value={password} onChange={e => setPassword(e.target.value)} required style={{ width: '100%', boxSizing: 'border-box', padding: '10px 12px', borderRadius: '8px', border: '1px solid var(--color-border)', background: 'var(--color-ground)', color: 'var(--color-ink)' }} />
            </div>
            {errorMsg && <div style={{ color: '#d93025', fontSize: '13px', marginBottom: '16px', padding: '8px 12px', background: 'rgba(217, 48, 37, 0.08)', borderRadius: '6px' }}>{errorMsg}</div>}
            <button type="submit" disabled={isSubmitting} style={{ width: '100%', padding: '10px', borderRadius: '8px', border: 'none', background: 'var(--color-teal)', color: 'white', fontWeight: 500, cursor: isSubmitting ? 'default' : 'pointer', transition: 'background 0.2s' }}>
              {isSubmitting ? 'Logging in...' : 'Log in'}
            </button>
          </form>
        ) : (
          <form onSubmit={handleRegister}>
            <h2 style={{ margin: '0 0 20px', color: 'var(--color-ink)', fontSize: '18px' }}>Request Access</h2>
            <div style={{ marginBottom: '14px' }}>
              <label style={{ display: 'block', fontSize: '13px', marginBottom: '6px', color: 'var(--color-ink-muted)' }}>Email</label>
              <input type="email" value={regEmail} onChange={e => setRegEmail(e.target.value)} required placeholder="name@company.com" style={{ width: '100%', boxSizing: 'border-box', padding: '10px 12px', borderRadius: '8px', border: '1px solid var(--color-border)', background: 'var(--color-ground)', color: 'var(--color-ink)' }} />
            </div>
            <div style={{ marginBottom: '14px' }}>
              <label style={{ display: 'block', fontSize: '13px', marginBottom: '6px', color: 'var(--color-ink-muted)' }}>Password</label>
              <input type="password" value={regPassword} onChange={e => setRegPassword(e.target.value)} required minLength={6} placeholder="Min 6 characters" style={{ width: '100%', boxSizing: 'border-box', padding: '10px 12px', borderRadius: '8px', border: '1px solid var(--color-border)', background: 'var(--color-ground)', color: 'var(--color-ink)' }} />
            </div>
            <div style={{ marginBottom: '14px' }}>
              <label style={{ display: 'block', fontSize: '13px', marginBottom: '6px', color: 'var(--color-ink-muted)' }}>Requested Department (optional)</label>
              <input type="text" value={regDept} onChange={e => setRegDept(e.target.value)} placeholder="e.g. SALES, FINANCE, ENGINEERING" style={{ width: '100%', boxSizing: 'border-box', padding: '10px 12px', borderRadius: '8px', border: '1px solid var(--color-border)', background: 'var(--color-ground)', color: 'var(--color-ink)' }} />
            </div>
            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', fontSize: '13px', marginBottom: '6px', color: 'var(--color-ink-muted)' }}>Reason for Access</label>
              <input type="text" value={regReason} onChange={e => setRegReason(e.target.value)} placeholder="Project or team purpose" style={{ width: '100%', boxSizing: 'border-box', padding: '10px 12px', borderRadius: '8px', border: '1px solid var(--color-border)', background: 'var(--color-ground)', color: 'var(--color-ink)' }} />
            </div>
            {errorMsg && <div style={{ color: '#d93025', fontSize: '13px', marginBottom: '16px', padding: '8px 12px', background: 'rgba(217, 48, 37, 0.08)', borderRadius: '6px' }}>{errorMsg}</div>}
            {successMsg && <div style={{ color: '#0f766e', fontSize: '13px', marginBottom: '16px', padding: '8px 12px', background: 'rgba(15, 118, 110, 0.1)', borderRadius: '6px' }}>{successMsg}</div>}
            <button type="submit" disabled={isSubmitting} style={{ width: '100%', padding: '10px', borderRadius: '8px', border: 'none', background: 'var(--color-teal)', color: 'white', fontWeight: 500, cursor: isSubmitting ? 'default' : 'pointer', transition: 'background 0.2s' }}>
              {isSubmitting ? 'Submitting...' : 'Submit Registration'}
            </button>
          </form>
        )}

      </div>
    </div>
  )
}
