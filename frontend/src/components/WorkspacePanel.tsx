import React, { useState } from 'react'
import { ConversationItem } from '../api/chat'
import { WorkspaceItem } from '../api/workspaces'

export interface WorkspacePanelProps {
  workspaces: WorkspaceItem[]
  activeWorkspaceId?: string | null
  onSelectWorkspace?: (wsId: string) => void
  onOpenWorkspaceSettings?: (ws: WorkspaceItem) => void
  conversations?: ConversationItem[]
  activeConversationId?: string | null
  onSelectConversation?: (convId: string, workspaceId?: string | null) => void
  onNewChat?: (workspaceId?: string | null) => void
  onDeleteConversation?: (e: React.MouseEvent, convId: string) => void
  loadingConversations?: boolean
}

export default function WorkspacePanel({
  workspaces = [],
  activeWorkspaceId = null,
  onSelectWorkspace,
  onOpenWorkspaceSettings,
  conversations = [],
  activeConversationId = null,
  onSelectConversation,
  onNewChat,
  onDeleteConversation,
  loadingConversations = false,
}: WorkspacePanelProps) {
  const [wsSearch, setWsSearch] = useState('')
  const [threadSearch, setThreadSearch] = useState('')

  // --- Workspace filtering ---
  const wsQ = wsSearch.toLowerCase().trim()
  const filteredWorkspaces = workspaces.filter((ws) => {
    if (!wsQ) return true
    return (
      ws.name.toLowerCase().includes(wsQ) ||
      ws.code.toLowerCase().includes(wsQ) ||
      (ws.department_name && ws.department_name.toLowerCase().includes(wsQ))
    )
  })

  // --- Threads for active workspace ---
  const activeWorkspaceThreads = conversations.filter(
    (c) => activeWorkspaceId && c.workspace_id === activeWorkspaceId
  )
  const unassignedThreads = conversations.filter(
    (c) => !c.workspace_id || !workspaces.some((w) => w.id === c.workspace_id)
  )
  const allThreads = activeWorkspaceId ? activeWorkspaceThreads : unassignedThreads

  const threadQ = threadSearch.toLowerCase().trim()
  const filteredThreads = allThreads.filter((t) => {
    if (!threadQ) return true
    return t.title.toLowerCase().includes(threadQ)
  })

  const activeWorkspace = workspaces.find((w) => w.id === activeWorkspaceId)

  const handleWsClick = (wsId: string) => {
    onSelectWorkspace && onSelectWorkspace(wsId)
  }

  const handleNewThreadClick = () => {
    onNewChat && onNewChat(activeWorkspaceId)
  }

  const handleThreadClick = (convId: string) => {
    onSelectConversation && onSelectConversation(convId, activeWorkspaceId)
  }

  return (
    <div
      style={{
        width: '260px',
        minWidth: '260px',
        height: '100vh',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
        background: 'var(--color-surface)',
        borderRight: '1px solid var(--color-border)',
        fontFamily: 'var(--font-sans)',
      }}
    >
      {/* ===== WORKSPACES SECTION ===== */}
      <div style={{ padding: '16px 14px 8px', borderBottom: '1px solid var(--color-border)' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
          <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--color-ink)', letterSpacing: '-0.01em' }}>
            Workspaces
          </span>
        </div>

        {/* Search Workspaces */}
        <div style={{ position: 'relative', marginBottom: '8px' }}>
          <input
            type="text"
            placeholder="Search workspaces..."
            value={wsSearch}
            onChange={(e) => setWsSearch(e.target.value)}
            style={{
              width: '100%',
              padding: '7px 10px 7px 30px',
              borderRadius: '8px',
              border: '1px solid var(--color-border)',
              background: 'var(--color-ground)',
              fontSize: '12px',
              color: 'var(--color-ink)',
              boxSizing: 'border-box',
              outline: 'none',
            }}
          />
          <svg
            width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="var(--color-ink-muted)" strokeWidth="2" strokeLinecap="round"
            style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none' }}
          >
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          {wsSearch && (
            <button
              onClick={() => setWsSearch('')}
              style={{ position: 'absolute', right: '8px', top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', color: 'var(--color-ink-muted)', cursor: 'pointer', fontSize: '11px', padding: 0 }}
            >✕</button>
          )}
        </div>

        {/* Workspace List */}
        <div style={{ maxHeight: '200px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '2px' }}>
          {filteredWorkspaces.length === 0 ? (
            <div style={{ padding: '12px 8px', fontSize: '12px', color: 'var(--color-ink-muted)', fontStyle: 'italic' }}>No workspaces found.</div>
          ) : (
            filteredWorkspaces.map((ws) => {
              const isActive = ws.id === activeWorkspaceId
              return (
                <div
                  key={ws.id}
                  onClick={() => handleWsClick(ws.id)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '8px 10px',
                    borderRadius: '8px',
                    background: isActive ? 'var(--color-teal-light)' : 'transparent',
                    border: isActive ? '1px solid var(--color-teal)' : '1px solid transparent',
                    cursor: 'pointer',
                    transition: 'all 0.12s ease',
                  }}
                  onMouseEnter={(e) => {
                    if (!isActive) e.currentTarget.style.background = 'var(--color-ground)'
                  }}
                  onMouseLeave={(e) => {
                    if (!isActive) e.currentTarget.style.background = 'transparent'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flex: 1, minWidth: 0 }}>
                    {/* Workspace icon */}
                    <div
                      style={{
                        width: '28px',
                        height: '28px',
                        borderRadius: '6px',
                        background: isActive ? 'var(--color-teal)' : 'var(--color-panel)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        flexShrink: 0,
                        color: isActive ? 'white' : 'var(--color-ink-muted)',
                        fontSize: '12px',
                        fontWeight: 600,
                      }}
                    >
                      {ws.name.charAt(0).toUpperCase()}
                    </div>
                    <div style={{ minWidth: 0, overflow: 'hidden' }}>
                      <div
                        style={{
                          fontSize: '13px',
                          fontWeight: isActive ? 600 : 500,
                          color: isActive ? 'var(--color-teal-dim)' : 'var(--color-ink)',
                          whiteSpace: 'nowrap',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                        }}
                        title={`${ws.name} (${ws.code})`}
                      >
                        {ws.name}
                      </div>
                      {ws.department_code && (
                        <div style={{ fontSize: '10px', color: 'var(--color-ink-muted)', fontFamily: 'var(--font-mono)', marginTop: '1px' }}>
                          {ws.department_code}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Settings gear */}
                  {onOpenWorkspaceSettings && isActive && (
                    <button
                      onClick={(e) => { e.stopPropagation(); onOpenWorkspaceSettings(ws) }}
                      title="Workspace Settings"
                      style={{
                        background: 'transparent',
                        border: 'none',
                        color: 'var(--color-ink-muted)',
                        padding: '2px',
                        cursor: 'pointer',
                        fontSize: '14px',
                        borderRadius: '4px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                      }}
                      onMouseEnter={(e) => (e.currentTarget.style.color = 'var(--color-teal)')}
                      onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--color-ink-muted)')}
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="12" cy="12" r="3" />
                        <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-2 2 2 2 0 01-2-2v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 01-2-2 2 2 0 012-2h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 010-2.83 2 2 0 012.83 0l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 012-2 2 2 0 012 2v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 0 2 2 0 010 2.83l-.06.06a1.65 1.65 0 00-.33 1.82V9a1.65 1.65 0 001.51 1H21a2 2 0 012 2 2 2 0 01-2 2h-.09a1.65 1.65 0 00-1.51 1z" />
                      </svg>
                    </button>
                  )}
                </div>
              )
            })
          )}
        </div>
      </div>

      {/* ===== THREADS SECTION ===== */}
      <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', padding: '12px 14px 8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
          <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--color-ink)', letterSpacing: '-0.01em' }}>
            Threads
          </span>
          <button
            onClick={handleNewThreadClick}
            title="New Thread"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
              padding: '4px 10px',
              borderRadius: '6px',
              background: 'var(--color-teal)',
              border: 'none',
              color: 'white',
              fontSize: '11px',
              fontWeight: 600,
              cursor: 'pointer',
              transition: 'all 0.12s ease',
            }}
            onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--color-teal-dim)')}
            onMouseLeave={(e) => (e.currentTarget.style.background = 'var(--color-teal)')}
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
              <line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" />
            </svg>
            New Thread
          </button>
        </div>

        {/* Active Workspace Label */}
        {activeWorkspace && (
          <div style={{ fontSize: '10px', color: 'var(--color-teal)', fontFamily: 'var(--font-mono)', padding: '0 0 6px', letterSpacing: '0.04em' }}>
            {activeWorkspace.name} threads
          </div>
        )}

        {/* Search Threads */}
        <div style={{ position: 'relative', marginBottom: '8px' }}>
          <input
            type="text"
            placeholder="Search threads..."
            value={threadSearch}
            onChange={(e) => setThreadSearch(e.target.value)}
            style={{
              width: '100%',
              padding: '7px 10px 7px 30px',
              borderRadius: '8px',
              border: '1px solid var(--color-border)',
              background: 'var(--color-ground)',
              fontSize: '12px',
              color: 'var(--color-ink)',
              boxSizing: 'border-box',
              outline: 'none',
            }}
          />
          <svg
            width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="var(--color-ink-muted)" strokeWidth="2" strokeLinecap="round"
            style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none' }}
          >
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          {threadSearch && (
            <button
              onClick={() => setThreadSearch('')}
              style={{ position: 'absolute', right: '8px', top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', color: 'var(--color-ink-muted)', cursor: 'pointer', fontSize: '11px', padding: 0 }}
            >✕</button>
          )}
        </div>

        {/* Thread List */}
        <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '2px' }}>
          {loadingConversations ? (
            <div style={{ padding: '16px 8px', fontSize: '12px', color: 'var(--color-ink-muted)', textAlign: 'center' }}>Loading threads…</div>
          ) : filteredThreads.length === 0 ? (
            <div style={{ padding: '16px 8px', fontSize: '12px', color: 'var(--color-ink-muted)', textAlign: 'center', fontStyle: 'italic' }}>
              {activeWorkspaceId ? 'No threads in this workspace yet.' : 'No threads found.'}
            </div>
          ) : (
            filteredThreads.map((thread) => {
              const isActive = thread.id === activeConversationId
              return (
                <div
                  key={thread.id}
                  onClick={() => handleThreadClick(thread.id)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '8px 10px',
                    borderRadius: '8px',
                    background: isActive ? 'var(--color-teal-light)' : 'transparent',
                    cursor: 'pointer',
                    transition: 'all 0.1s ease',
                  }}
                  onMouseEnter={(e) => {
                    if (!isActive) e.currentTarget.style.background = 'var(--color-ground)'
                  }}
                  onMouseLeave={(e) => {
                    if (!isActive) e.currentTarget.style.background = 'transparent'
                  }}
                  title={thread.title}
                >
                  <div style={{ flex: 1, minWidth: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={isActive ? 'var(--color-teal)' : 'var(--color-ink-muted)'} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
                      <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
                    </svg>
                    <span
                      style={{
                        fontSize: '12px',
                        fontWeight: isActive ? 600 : 400,
                        color: isActive ? 'var(--color-teal-dim)' : 'var(--color-ink)',
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                      }}
                    >
                      {thread.title}
                    </span>
                  </div>

                  {/* Delete button */}
                  {onDeleteConversation && (
                    <button
                      onClick={(e) => onDeleteConversation(e, thread.id)}
                      title="Delete thread"
                      style={{
                        background: 'transparent',
                        border: 'none',
                        color: 'var(--color-ink-muted)',
                        padding: '2px',
                        cursor: 'pointer',
                        opacity: 0,
                        fontSize: '11px',
                        flexShrink: 0,
                        transition: 'opacity 0.15s',
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.opacity = '1'
                        e.currentTarget.style.color = '#ef4444'
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.opacity = '0'
                        e.currentTarget.style.color = 'var(--color-ink-muted)'
                      }}
                    >
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                        <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
                      </svg>
                    </button>
                  )}
                </div>
              )
            })
          )}

          {/* Unassigned threads when a workspace is selected */}
          {activeWorkspaceId && unassignedThreads.length > 0 && (
            <div style={{ marginTop: '10px', borderTop: '1px dashed var(--color-border)', paddingTop: '8px' }}>
              <div style={{ fontSize: '10px', color: 'var(--color-ink-muted)', fontFamily: 'var(--font-mono)', letterSpacing: '0.06em', textTransform: 'uppercase', padding: '0 4px 4px', fontWeight: 600 }}>
                GENERAL ({unassignedThreads.length})
              </div>
              {unassignedThreads.map((thread) => {
                const isActive = thread.id === activeConversationId
                return (
                  <div
                    key={thread.id}
                    onClick={() => onSelectConversation && onSelectConversation(thread.id, null)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      padding: '6px 10px',
                      borderRadius: '6px',
                      background: isActive ? 'var(--color-teal-light)' : 'transparent',
                      cursor: 'pointer',
                      gap: '8px',
                      transition: 'all 0.1s ease',
                    }}
                    onMouseEnter={(e) => {
                      if (!isActive) e.currentTarget.style.background = 'var(--color-ground)'
                    }}
                    onMouseLeave={(e) => {
                      if (!isActive) e.currentTarget.style.background = 'transparent'
                    }}
                    title={thread.title}
                  >
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke={isActive ? 'var(--color-teal)' : 'var(--color-ink-muted)'} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
                      <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
                    </svg>
                    <span style={{ fontSize: '11px', color: isActive ? 'var(--color-teal-dim)' : 'var(--color-ink)', fontWeight: isActive ? 600 : 400, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {thread.title}
                    </span>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
