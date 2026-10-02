import React from 'react'
import { WorkspaceItem } from '../api/workspaces'

export type NavItem =
  | 'chat'
  | 'sources'
  | 'settings'
  | 'users'
  | 'departments'
  | 'requests'
  | 'my-requests'
  | 'kb'
  | 'analytics'
  | 'access'
  | 'workspaces'

export interface SidebarProps {
  collapsed: boolean
  onToggleCollapse: () => void
  role: 'ADMIN' | 'USER' | string
  userName: string
  userEmail?: string
  onLogout: () => void
  activeItem: NavItem
  onSelect: (item: NavItem) => void
  pendingRequestsCount?: number
  // Backward compat — workspace/thread data is no longer rendered in sidebar
  workspaces?: WorkspaceItem[]
  activeWorkspaceId?: string | null
  onSelectWorkspace?: (wsId: string) => void
  onOpenWorkspaceSettings?: (ws: WorkspaceItem) => void
  conversations?: any[]
  activeConversationId?: string | null
  onSelectConversation?: (id: string, workspaceId?: string | null) => void
  onNewChat?: (workspaceId?: string | null) => void
  onDeleteConversation?: (e: React.MouseEvent, id: string) => void
  loadingConversations?: boolean
}

export default function Sidebar({
  collapsed,
  onToggleCollapse,
  role,
  userName,
  userEmail,
  onLogout,
  activeItem,
  onSelect,
  pendingRequestsCount = 0,
}: SidebarProps) {
  const isAdmin = role === 'ADMIN'

  /* ─────────────────────────── COLLAPSED (Icon Rail, ~56px) ─────────────────────────── */
  if (collapsed) {
    return (
      <aside
        style={{
          width: '56px',
          minWidth: '56px',
          height: '100vh',
          background: 'var(--color-surface)',
          borderRight: '1px solid var(--color-border)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '14px 0 10px',
          fontFamily: 'var(--font-sans)',
          zIndex: 20,
          flexShrink: 0,
          overflow: 'hidden',
          userSelect: 'none',
          transition: 'width 0.2s ease',
        }}
      >
        {/* Top: Brand + Icons */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '2px', width: '100%' }}>
          {/* Brand */}
          <div
            style={{
              width: '34px', height: '34px', borderRadius: '9px',
              background: 'var(--color-teal)', display: 'flex', alignItems: 'center', justifyContent: 'center',
              color: 'white', fontWeight: 700, fontSize: '15px', marginBottom: '10px', cursor: 'pointer',
              fontFamily: 'var(--font-serif)',
            }}
            onClick={onToggleCollapse}
            title="Expand sidebar"
          >
            R
          </div>

          <Sep />

          <RailBtn active={activeItem === 'chat'} title="Chat" onClick={() => onSelect('chat')}>
            <IconChat />
          </RailBtn>
          <RailBtn active={activeItem === 'sources'} title="Documents" onClick={() => onSelect('sources')}>
            <IconDoc />
          </RailBtn>
          <RailBtn active={activeItem === 'kb'} title="Knowledge" onClick={() => onSelect('kb')}>
            <IconKB />
          </RailBtn>

          {!isAdmin && (
            <>
              <RailBtn active={activeItem === 'access'} title="My Access" onClick={() => onSelect('access')}>
                <IconLock />
              </RailBtn>
              <RailBtn active={activeItem === 'my-requests'} title="My Requests" onClick={() => onSelect('my-requests')}>
                <IconRequest />
              </RailBtn>
            </>
          )}

          {isAdmin && (
            <>
              <Sep />
              <RailBtn active={activeItem === 'users'} title="Users" onClick={() => onSelect('users')}>
                <IconUsers />
              </RailBtn>
              <RailBtn active={activeItem === 'departments'} title="Departments" onClick={() => onSelect('departments')}>
                <IconDept />
              </RailBtn>
              <div style={{ position: 'relative' }}>
                <RailBtn active={activeItem === 'requests'} title="Access Requests" onClick={() => onSelect('requests')}>
                  <IconAccessReq />
                </RailBtn>
                {pendingRequestsCount > 0 && <BadgeDot />}
              </div>
              <RailBtn active={activeItem === 'workspaces'} title="Manage Workspaces" onClick={() => onSelect('workspaces')}>
                <IconWorkspace />
              </RailBtn>
            </>
          )}
        </div>

        {/* Bottom: avatar + logout */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '6px' }}>
          <Avatar name={userName} size={30} />
          <button onClick={onLogout} title="Sign out" style={logoutBtnStyle}>
            <IconLogout />
          </button>
        </div>
      </aside>
    )
  }

  /* ─────────────────────────── EXPANDED (~200px) ─────────────────────────── */
  return (
    <aside
      style={{
        width: '200px',
        minWidth: '200px',
        height: '100vh',
        background: 'var(--color-surface)',
        borderRight: '1px solid var(--color-border)',
        display: 'flex',
        flexDirection: 'column',
        fontFamily: 'var(--font-sans)',
        color: 'var(--color-ink)',
        zIndex: 20,
        flexShrink: 0,
        overflow: 'hidden',
        userSelect: 'none',
        transition: 'width 0.2s ease',
      }}
    >
      {/* Header: Brand + collapse */}
      <div
        style={{
          padding: '14px 12px 10px',
          borderBottom: '1px solid var(--color-border)',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '9px' }}>
          <div
            style={{
              width: '30px', height: '30px', borderRadius: '8px',
              background: 'var(--color-teal)', display: 'flex', alignItems: 'center', justifyContent: 'center',
              color: 'white', fontWeight: 700, fontSize: '13px', flexShrink: 0,
              fontFamily: 'var(--font-serif)',
            }}
          >
            R
          </div>
          <div>
            <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--color-ink)', letterSpacing: '-0.01em' }}>RAG Platform</div>
            <div style={{ fontSize: '9px', color: 'var(--color-ink-muted)', fontFamily: 'var(--font-mono)' }}>Enterprise AI</div>
          </div>
        </div>
        <button
          onClick={onToggleCollapse}
          title="Collapse sidebar"
          style={{
            width: '26px', height: '26px', borderRadius: '6px',
            background: 'var(--color-ground)', border: '1px solid var(--color-border)',
            color: 'var(--color-ink-muted)', cursor: 'pointer',
            display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
          }}
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="15 18 9 12 15 6" />
          </svg>
        </button>
      </div>

      {/* Nav */}
      <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', padding: '10px 8px' }}>
        <SectionLabel>MAIN</SectionLabel>
        <NavBtn active={activeItem === 'chat'} onClick={() => onSelect('chat')}><IconChat /> Chat</NavBtn>
        <NavBtn active={activeItem === 'sources'} onClick={() => onSelect('sources')}><IconDoc /> Documents</NavBtn>
        <NavBtn active={activeItem === 'kb'} onClick={() => onSelect('kb')}><IconKB /> Knowledge</NavBtn>

        {!isAdmin && (
          <>
            <SectionLabel>ACCESS</SectionLabel>
            <NavBtn active={activeItem === 'access'} onClick={() => onSelect('access')}><IconLock /> My Access</NavBtn>
            <NavBtn active={activeItem === 'my-requests'} onClick={() => onSelect('my-requests')}><IconRequest /> My Requests</NavBtn>
          </>
        )}

        {isAdmin && (
          <>
            <SectionLabel>ADMIN</SectionLabel>
            <NavBtn active={activeItem === 'users'} onClick={() => onSelect('users')}><IconUsers /> Users</NavBtn>
            <NavBtn active={activeItem === 'departments'} onClick={() => onSelect('departments')}><IconDept /> Departments</NavBtn>
            <NavBtn active={activeItem === 'requests'} onClick={() => onSelect('requests')}>
              <IconAccessReq /> Requests
              {pendingRequestsCount > 0 && (
                <span style={{ marginLeft: 'auto', background: '#f59e0b', color: 'white', fontSize: '9px', fontWeight: 700, padding: '1px 5px', borderRadius: '10px', lineHeight: 1.4 }}>
                  {pendingRequestsCount}
                </span>
              )}
            </NavBtn>
            <NavBtn active={activeItem === 'workspaces'} onClick={() => onSelect('workspaces')}><IconWorkspace /> Workspaces</NavBtn>
          </>
        )}
      </div>

      {/* Bottom: user */}
      <div
        style={{
          borderTop: '1px solid var(--color-border)',
          padding: '10px 10px',
          display: 'flex', alignItems: 'center', gap: '8px',
        }}
      >
        <Avatar name={userName} size={28} />
        <div style={{ flex: 1, minWidth: 0, overflow: 'hidden' }}>
          <div style={{ fontSize: '12px', fontWeight: 500, color: 'var(--color-ink)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {userName || 'User'}
          </div>
          {userEmail && (
            <div style={{ fontSize: '9px', color: 'var(--color-ink-muted)', fontFamily: 'var(--font-mono)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
              {userEmail}
            </div>
          )}
        </div>
        <button onClick={onLogout} title="Sign out" style={logoutBtnStyle}>
          <IconLogout />
        </button>
      </div>
    </aside>
  )
}

/* ═══════════════════════════════════════ SUB‑COMPONENTS ═══════════════════════════════════════ */

const logoutBtnStyle: React.CSSProperties = {
  padding: '4px', background: 'transparent', border: 'none', borderRadius: '4px',
  color: 'var(--color-ink-muted)', cursor: 'pointer', flexShrink: 0,
  display: 'flex', alignItems: 'center',
}

function Sep() {
  return <div style={{ width: '24px', height: '1px', background: 'var(--color-border)', margin: '4px 0' }} />
}

function BadgeDot() {
  return (
    <div style={{ position: 'absolute', top: '6px', right: '6px', width: '7px', height: '7px', borderRadius: '50%', background: '#f59e0b', border: '1px solid var(--color-surface)' }} />
  )
}

function Avatar({ name, size = 28 }: { name: string; size?: number }) {
  return (
    <div
      style={{
        width: `${size}px`, height: `${size}px`, borderRadius: '50%',
        background: 'var(--color-teal)', display: 'flex', alignItems: 'center', justifyContent: 'center',
        color: 'white', fontWeight: 600, fontSize: `${Math.round(size * 0.4)}px`, flexShrink: 0,
      }}
    >
      {name ? name.charAt(0).toUpperCase() : 'U'}
    </div>
  )
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      fontSize: '9px', fontFamily: 'var(--font-mono)', letterSpacing: '0.1em',
      color: 'var(--color-ink-muted)', textTransform: 'uppercase',
      padding: '6px 8px 3px', fontWeight: 600,
    }}>
      {children}
    </div>
  )
}

function NavBtn({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      style={{
        display: 'flex', alignItems: 'center', gap: '8px', width: '100%',
        padding: '7px 8px', background: active ? 'var(--color-teal-light)' : 'transparent',
        border: active ? '1px solid var(--color-teal)' : '1px solid transparent',
        borderRadius: '7px', color: active ? 'var(--color-teal-dim)' : 'var(--color-ink)',
        fontSize: '12px', fontWeight: active ? 600 : 400, cursor: 'pointer',
        textAlign: 'left', transition: 'all 0.1s ease', marginBottom: '1px',
      }}
      onMouseEnter={(e) => { if (!active) { e.currentTarget.style.background = 'var(--color-ground)' } }}
      onMouseLeave={(e) => { if (!active) { e.currentTarget.style.background = 'transparent' } }}
    >
      {children}
    </button>
  )
}

function RailBtn({ active, title, onClick, children }: { active: boolean; title: string; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick} title={title}
      style={{
        width: '38px', height: '38px', borderRadius: '9px',
        background: active ? 'var(--color-teal-light)' : 'transparent',
        border: active ? '1px solid var(--color-teal)' : '1px solid transparent',
        color: active ? 'var(--color-teal)' : 'var(--color-ink-muted)', cursor: 'pointer',
        display: 'flex', alignItems: 'center', justifyContent: 'center', transition: 'all 0.1s ease',
      }}
      onMouseEnter={(e) => { if (!active) { e.currentTarget.style.background = 'var(--color-ground)'; e.currentTarget.style.color = 'var(--color-ink)' } }}
      onMouseLeave={(e) => { if (!active) { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = 'var(--color-ink-muted)' } }}
    >
      {children}
    </button>
  )
}

/* ═══════════════ ICONS (16×16 SVG) ═══════════════ */

function IconChat() {
  return (<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" /></svg>)
}
function IconDoc() {
  return (<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" /><polyline points="14 2 14 8 20 8" /><line x1="16" y1="13" x2="8" y2="13" /><line x1="16" y1="17" x2="8" y2="17" /></svg>)
}
function IconKB() {
  return (<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="3" /><circle cx="19" cy="5" r="2" /><circle cx="5" cy="5" r="2" /><circle cx="5" cy="19" r="2" /><circle cx="19" cy="19" r="2" /><line x1="14.5" y1="9.5" x2="17.5" y2="6.5" /><line x1="9.5" y1="9.5" x2="6.5" y2="6.5" /><line x1="9.5" y1="14.5" x2="6.5" y2="17.5" /><line x1="14.5" y1="14.5" x2="17.5" y2="17.5" /></svg>)
}
function IconLock() {
  return (<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="11" width="18" height="11" rx="2" ry="2" /><path d="M7 11V7a5 5 0 0110 0v4" /></svg>)
}
function IconRequest() {
  return (<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" /><polyline points="14 2 14 8 20 8" /><line x1="12" y1="18" x2="12" y2="12" /><line x1="9" y1="15" x2="15" y2="15" /></svg>)
}
function IconUsers() {
  return (<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2" /><circle cx="9" cy="7" r="4" /><path d="M23 21v-2a4 4 0 00-3-3.87" /><path d="M16 3.13a4 4 0 010 7.75" /></svg>)
}
function IconDept() {
  return (<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z" /><polyline points="9 22 9 12 15 12 15 22" /></svg>)
}
function IconAccessReq() {
  return (<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M16 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2" /><circle cx="8.5" cy="7" r="4" /><line x1="20" y1="8" x2="20" y2="14" /><line x1="23" y1="11" x2="17" y2="11" /></svg>)
}
function IconWorkspace() {
  return (<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><rect x="2" y="3" width="20" height="14" rx="2" ry="2" /><line x1="8" y1="21" x2="16" y2="21" /><line x1="12" y1="17" x2="12" y2="21" /></svg>)
}
function IconLogout() {
  return (<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4" /><polyline points="16 17 21 12 16 7" /><line x1="21" y1="12" x2="9" y2="12" /></svg>)
}
