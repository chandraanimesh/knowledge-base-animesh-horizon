import os

base_dir = r"c:\ragDemo\ragdemo1\frontend\src"

def write_file(path, content):
    full_path = os.path.join(base_dir, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(content)

login_tsx = """import { useState } from 'react'
import { login } from '../api/auth'

export default function Login({ onLogin }: { onLogin: (token: string) => void }) {
  const [username, setUsername] = useState('admin@ragdemo.com')
  const [password, setPassword] = useState('AdminPassword123!')
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
"""

rag_tsx = """import { useState, useRef, useEffect, ChangeEvent } from 'react'
import { askQuestion, GenerateRequest, RetrievedChunk } from '../api/chat'
import { uploadDocument } from '../api/documents'

type Role = 'user' | 'assistant'
interface Source { title: string; excerpt: string; }
interface Message { id: number; role: Role; text: string; sources?: Source[]; isTyping?: boolean; }

function SourceCard({ source }: { source: Source }) {
  return (
    <div style={{ background: 'var(--color-teal-light)', borderLeft: '2px solid var(--color-teal)', padding: '10px 14px', borderRadius: '4px', marginTop: '8px' }}>
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', letterSpacing: '0.08em', color: 'var(--color-teal)', marginBottom: '4px', textTransform: 'uppercase' }}>{source.title}</div>
      <div style={{ fontSize: '13px', color: 'var(--color-ink-muted)', lineHeight: 1.5, fontStyle: 'italic' }}>"{source.excerpt}"</div>
    </div>
  )
}

const THINKING_STEPS = [
  { label: 'Doot is thinking…', icon: '◌' },
  { label: 'Using web search', icon: '⌕' },
  { label: 'Reading sources', icon: '⊡' },
  { label: 'Answer is about to come', icon: '◎' },
]

function TypingIndicator() {
  const [step, setStep] = useState(0)
  const [visible, setVisible] = useState(true)

  useEffect(() => {
    const interval = setInterval(() => {
      setVisible(false)
      setTimeout(() => { setStep((s) => (s + 1) % THINKING_STEPS.length); setVisible(true) }, 200)
    }, 1400)
    return () => clearInterval(interval)
  }, [])

  const current = THINKING_STEPS[step]

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', opacity: visible ? 1 : 0, transition: 'opacity 0.2s ease' }}>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '15px', color: 'var(--color-teal)', animation: 'spin 1.8s linear infinite', display: 'inline-block' }}>{current.icon}</span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--color-teal)', letterSpacing: '0.04em' }}>{current.label}</span>
      </div>
      <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
        {THINKING_STEPS.map((_, i) => (
          <div key={i} style={{ width: i === step ? '18px' : '5px', height: '5px', borderRadius: '3px', background: i === step ? 'var(--color-teal)' : 'var(--color-border)', transition: 'all 0.3s ease' }} />
        ))}
      </div>
      <style>{`@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`}</style>
    </div>
  )
}

function AssistantMessage({ msg }: { msg: Message }) {
  return (
    <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-start' }}>
      <div style={{ width: '28px', height: '28px', borderRadius: '50%', background: 'var(--color-teal)', flexShrink: 0, marginTop: '3px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: 'rgba(255,255,255,0.5)' }} />
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', letterSpacing: '0.1em', color: 'var(--color-teal)', marginBottom: '8px', textTransform: 'uppercase' }}>Doot</div>
        {msg.isTyping ? <TypingIndicator /> : (
          <>
            <p style={{ margin: 0, fontFamily: "'Segoe UI', system-ui, sans-serif", fontSize: '17px', lineHeight: 1.7, color: 'var(--color-ink)', whiteSpace: 'pre-wrap' }}>{msg.text}</p>
            {msg.sources && msg.sources.length > 0 && (
              <div style={{ marginTop: '12px' }}>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', letterSpacing: '0.08em', color: 'var(--color-ink-muted)', textTransform: 'uppercase', marginBottom: '6px' }}>{msg.sources.length} source{msg.sources.length !== 1 ? 's' : ''} retrieved</div>
                {msg.sources.map((s, i) => <SourceCard key={i} source={s} />)}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

function UserMessage({ msg }: { msg: Message }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
      <div style={{ maxWidth: '68%', background: 'var(--color-user-bubble)', color: 'var(--color-user-text)', borderRadius: '14px 14px 3px 14px', padding: '12px 18px', fontSize: '15px', lineHeight: 1.6, fontFamily: 'var(--font-sans)', fontWeight: 400, whiteSpace: 'pre-wrap' }}>{msg.text}</div>
    </div>
  )
}

export default function RAG({ token, userName, allowedUploadDepts }: { token: string, userName: string, allowedUploadDepts: string[] }) {
  const [messages, setMessages] = useState<Message[]>([
    { id: 0, role: 'assistant', text: 'Hello. I\\'m Doot — a research assistant grounded in your document library. Ask me anything, and I\\'ll retrieve relevant sources before answering.' }
  ])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [topK, setTopK] = useState(5)
  const [rerank, setRerank] = useState(false)
  
  const [department, setDepartment] = useState(allowedUploadDepts.length > 0 ? allowedUploadDepts[0] : '')
  const [uploading, setUploading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState<string | null>(null)

  const bottomRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const idRef = useRef(1)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  function autoResize() {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 160) + 'px'
  }

  function triggerUpload() { fileInputRef.current?.click() }

  async function handleUpload(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file || !token || !department) return
    setUploading(true)
    setUploadStatus('Document is being processed… Queries will be enabled when ingestion completes.')
    try {
      const res = await uploadDocument(file, department, token)
      setUploadStatus(res.skipped ? `Document ready. ${res.file_name} was already indexed.` : `Document ready. ${res.file_name} indexed with ${res.chunks_created} chunks.`)
    } catch (err: any) {
      setUploadStatus(`Upload failed: ${err.message}`)
    } finally {
      setUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  async function send(text: string) {
    if (!text.trim() || isLoading || uploading || !token) return
    const userMsg: Message = { id: idRef.current++, role: 'user', text: text.trim() }
    const typingMsg: Message = { id: idRef.current++, role: 'assistant', text: '', isTyping: true }
    setMessages((m) => [...m, userMsg, typingMsg])
    setInput('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
    setIsLoading(true)
    try {
      const res = await askQuestion({ query: text.trim(), top_k: topK, rerank }, token)
      const mappedSources: Source[] = res.retrieved_chunks.map((c: RetrievedChunk) => ({
        title: `${c.file_name}${c.page_number ? ` (Page ${c.page_number})` : ''}`,
        excerpt: c.text
      }))
      setMessages((m) => m.map((msg) => msg.id === typingMsg.id ? { ...msg, text: res.answer, sources: mappedSources, isTyping: false } : msg))
    } catch (err: any) {
      setMessages((m) => m.map((msg) => msg.id === typingMsg.id ? { ...msg, text: `Error: ${err.message}`, isTyping: false } : msg))
    } finally {
      setIsLoading(false)
    }
  }

  function handleKey(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(input) }
  }

  const isInputDisabled = isLoading || uploading

  return (
    <div style={{ minHeight: '100dvh', display: 'grid', gridTemplateRows: 'auto 1fr auto', background: 'var(--color-ground)' }}>
      <header style={{ borderBottom: '1px solid var(--color-border)', padding: '0 32px', height: '56px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: 'var(--color-surface)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{ width: '22px', height: '22px', borderRadius: '50%', background: 'var(--color-teal)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'rgba(255,255,255,0.5)' }} />
          </div>
          <span style={{ fontFamily: 'var(--font-serif)', fontSize: '18px', color: 'var(--color-ink)', letterSpacing: '-0.01em' }}>{userName}'s</span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--color-teal)', background: 'var(--color-teal-light)', padding: '2px 8px', borderRadius: '3px', letterSpacing: '0.06em', textTransform: 'uppercase' }}>DOOT</span>
        </div>
        <div style={{ display: 'flex', gap: '20px', alignItems: 'center' }}>
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            <span style={{ fontSize: '11px', color: 'var(--color-ink-muted)', fontFamily: 'var(--font-mono)' }}>Top K:</span>
            <input type="number" value={topK} onChange={e => setTopK(parseInt(e.target.value) || 1)} min="1" max="20" style={{ width: '40px', background: 'transparent', border: '1px solid var(--color-border)', borderRadius: '4px', fontSize: '11px', textAlign: 'center', padding: '2px' }} />
          </div>
          <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
            <input type="checkbox" id="rerank" checked={rerank} onChange={e => setRerank(e.target.checked)} />
            <label htmlFor="rerank" style={{ fontSize: '11px', color: 'var(--color-ink-muted)', fontFamily: 'var(--font-mono)' }}>Rerank</label>
          </div>
        </div>
      </header>

      <main style={{ overflowY: 'auto', padding: '40px 0' }}>
        <div style={{ maxWidth: '680px', margin: '0 auto', padding: '0 24px', display: 'flex', flexDirection: 'column', gap: '32px' }}>
          {messages.map((msg) => msg.role === 'assistant' ? <AssistantMessage key={msg.id} msg={msg} /> : <UserMessage key={msg.id} msg={msg} />)}
          <div ref={bottomRef} />
        </div>
      </main>

      <footer style={{ borderTop: '1px solid var(--color-border)', background: 'var(--color-surface)', padding: '16px 24px 24px' }}>
        <div style={{ maxWidth: '680px', margin: '0 auto' }}>
          {uploadStatus && <div style={{ marginBottom: '12px', fontSize: '12px', color: uploading ? 'var(--color-teal)' : 'var(--color-ink-muted)', fontFamily: 'var(--font-mono)', textAlign: 'center' }}>{uploadStatus}</div>}
          <div style={{ display: 'flex', alignItems: 'flex-end', gap: '12px', background: isInputDisabled ? 'var(--color-panel)' : 'var(--color-ground)', border: '1px solid var(--color-border)', borderRadius: '12px', padding: '10px 12px 10px 16px', transition: 'border-color 0.15s, background 0.15s' }} onFocusCapture={(e) => { if(!isInputDisabled) e.currentTarget.style.borderColor = 'var(--color-teal)' }} onBlurCapture={(e) => (e.currentTarget.style.borderColor = 'var(--color-border)')}>
            <textarea ref={textareaRef} value={input} onChange={(e) => { setInput(e.target.value); autoResize() }} onKeyDown={handleKey} placeholder={uploading ? "Document is being processed..." : "Ask anything…"} disabled={isInputDisabled} rows={1} style={{ flex: 1, fontFamily: 'var(--font-sans)', fontSize: '15px', color: 'var(--color-ink)', background: 'transparent', border: 'none', resize: 'none', lineHeight: 1.6, maxHeight: '160px', overflowY: 'auto', opacity: isInputDisabled ? 0.6 : 1 }} />
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
              {allowedUploadDepts.length > 0 && (
                <>
                  <select value={department} onChange={e => setDepartment(e.target.value)} disabled={isInputDisabled} style={{ width: '80px', background: 'var(--color-surface)', border: '1px solid var(--color-border)', borderRadius: '6px', fontSize: '10px', padding: '6px', textTransform: 'uppercase' }}>
                    {allowedUploadDepts.map(d => <option key={d} value={d}>{d}</option>)}
                  </select>
                  <button onClick={triggerUpload} disabled={isInputDisabled} title="Upload Document" style={{ width: '36px', height: '36px', borderRadius: '8px', background: 'var(--color-panel)', border: '1px solid var(--color-border)', cursor: isInputDisabled ? 'default' : 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', opacity: isInputDisabled ? 0.6 : 1 }}>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
                  </button>
                  <input type="file" ref={fileInputRef} onChange={handleUpload} style={{ display: 'none' }} accept=".pdf,.txt,.md,.csv,.xlsx,.xls,.docx" />
                </>
              )}
              <button onClick={() => send(input)} disabled={!input.trim() || isInputDisabled} style={{ width: '36px', height: '36px', borderRadius: '8px', background: input.trim() && !isInputDisabled ? 'var(--color-teal)' : 'var(--color-panel)', border: 'none', cursor: input.trim() && !isInputDisabled ? 'pointer' : 'default', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, transition: 'background 0.15s' }}>
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M3 8H13M13 8L9 4M13 8L9 12" stroke={input.trim() && !isInputDisabled ? '#f5f5f0' : 'var(--color-ink-muted)'} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
              </button>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}
"""

user_dashboard = """import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { userApi } from '../api/user'
import RAG from '../components/RAG'

export default function UserDashboard({ token, onLogout }: { token: string, onLogout: () => void }) {
  const [userName, setUserName] = useState('')
  const [access, setAccess] = useState<Record<string, string[]> | null>(null)
  const navigate = useNavigate()

  useEffect(() => {
    async function loadData() {
      try {
        const user = await userApi.getMe(token)
        if (user.role === 'ADMIN') {
          navigate('/admin', { replace: true })
          return
        }
        const namePart = user.email.split('@')[0]
        setUserName(namePart.charAt(0).toUpperCase() + namePart.slice(1))
        
        const myAccess = await userApi.getMyAccess(token)
        setAccess(myAccess)
      } catch (e) {
        onLogout()
      }
    }
    loadData()
  }, [token, navigate, onLogout])

  if (!access) return <div>Loading access profile...</div>

  const allowedUploadDepts = Object.keys(access).filter(d => access[d].includes('UPLOAD') || access[d].includes('ALL'))

  return (
    <div style={{ display: 'flex', height: '100dvh' }}>
      <div style={{ width: '250px', background: 'var(--color-surface)', borderRight: '1px solid var(--color-border)', padding: '20px', display: 'flex', flexDirection: 'column' }}>
        <h2 style={{ fontSize: '18px', marginBottom: '20px' }}>My Access</h2>
        <div style={{ flex: 1, overflowY: 'auto', fontSize: '13px' }}>
          {Object.keys(access).length === 0 ? <p style={{color: 'var(--color-ink-muted)'}}>No department access.</p> : 
            Object.entries(access).map(([dept, perms]) => (
              <div key={dept} style={{ marginBottom: '16px' }}>
                <strong style={{ display: 'block', marginBottom: '4px', textTransform: 'uppercase', color: 'var(--color-teal)' }}>{dept}</strong>
                <ul style={{ listStyle: 'none', margin: 0, padding: 0 }}>
                  {perms.map(p => <li key={p} style={{ color: 'var(--color-ink-muted)' }}>✓ {p}</li>)}
                </ul>
              </div>
            ))
          }
        </div>
        <button onClick={onLogout} style={{ padding: '8px', background: 'transparent', border: '1px solid var(--color-border)', borderRadius: '4px', cursor: 'pointer', marginTop: 'auto' }}>Logout</button>
      </div>
      <div style={{ flex: 1, position: 'relative' }}>
        <RAG token={token} userName={userName} allowedUploadDepts={allowedUploadDepts} />
      </div>
    </div>
  )
}
"""

app_tsx = """import { useState } from 'react'
import { BrowserRouter, Routes, Route, Navigate, useNavigate } from 'react-router-dom'
import Login from './components/Login'
import UserDashboard from './pages/UserDashboard'
import AdminDashboard from './pages/AdminDashboard'

function AuthBoundary() {
  const [token, setToken] = useState<string | null>(localStorage.getItem('token'))
  const navigate = useNavigate()

  const handleLogin = (jwt: string) => {
    setToken(jwt)
    localStorage.setItem('token', jwt)
    navigate('/') // This root path handles redirection based on role inside the Dashboard components
  }

  const handleLogout = () => {
    setToken(null)
    localStorage.removeItem('token')
    navigate('/login', { replace: true })
  }

  return (
    <Routes>
      <Route path="/login" element={!token ? <Login onLogin={handleLogin} /> : <Navigate to="/" replace />} />
      
      {/* Protected Routes */}
      <Route path="/admin/*" element={token ? <AdminDashboard token={token} onLogout={handleLogout} /> : <Navigate to="/login" replace />} />
      <Route path="/app/*" element={token ? <UserDashboard token={token} onLogout={handleLogout} /> : <Navigate to="/login" replace />} />
      
      {/* Root redirection based on user's default entry. Admin/User components handle role check and redirect if wrong */}
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
"""

write_file('components/Login.tsx', login_tsx)
write_file('components/RAG.tsx', rag_tsx)
write_file('pages/UserDashboard.tsx', user_dashboard)
write_file('App.tsx', app_tsx)
print("React files generated successfully!")
