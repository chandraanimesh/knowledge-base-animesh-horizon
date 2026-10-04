import { useEffect, useState, useCallback } from 'react'
import { useNavigate, useLocation, matchPath } from 'react-router-dom'
import { userApi } from '../api/user'
import { departmentApi } from '../api/departments'
import {
  accessRequestApi,
  AccessRequestResponse,
  PermissionCode,
  AccessRequestStatus,
} from '../api/accessRequests'
import { DepartmentResponse } from '../api/admin'
import RAG from '../components/RAG'
import Sidebar, { NavItem } from '../components/Sidebar'
import {
  ConversationItem,
  listConversations,
  deleteConversation,
} from '../api/chat'
import { documentApi, DocumentListItem } from '../api/documents'
import { listWorkspaces, WorkspaceItem } from '../api/workspaces'
import WorkspaceProviderSettingsModal from '../components/WorkspaceProviderSettingsModal'
import WorkspacePanel from '../components/WorkspacePanel'
import AnythingLLMApp from '../components/AnythingLLMApp'
const ALL_PERMS: PermissionCode[] = ['VIEW', 'UPLOAD', 'EDIT', 'DELETE']

const PRESETS: Record<string, PermissionCode[]> = {
  'View Only': ['VIEW'],
  Contributor: ['VIEW', 'UPLOAD', 'EDIT'],
  'Full Access': ['VIEW', 'UPLOAD', 'EDIT', 'DELETE'],
}

function statusBadge(status: AccessRequestStatus | string) {
  switch (status) {
    case 'PENDING':
      return { bg: '#fff8e1', color: '#b45309', border: '#fde68a', label: 'Pending' }
    case 'APPROVED':
      return { bg: '#ecfdf5', color: '#047857', border: '#a7f3d0', label: 'Approved' }
    case 'REJECTED':
      return { bg: '#fef2f2', color: '#b91c1c', border: '#fecaca', label: 'Rejected' }
    case 'CANCELLED':
      return { bg: '#f3f4f6', color: '#4b5563', border: '#e5e7eb', label: 'Cancelled' }
    default:
      return { bg: '#f3f4f6', color: '#4b5563', border: '#e5e7eb', label: status }
  }
}

export default function UserDashboard({
  token,
  onLogout,
}: {
  token: string
  onLogout: () => void
}) {
  const [userName, setUserName] = useState('')
  const [userEmail, setUserEmail] = useState('')
  const [access, setAccess] = useState<Record<string, string[]> | null>(null)
  const [myRequests, setMyRequests] = useState<AccessRequestResponse[]>([])
  const [activeDepts, setActiveDepts] = useState<DepartmentResponse[]>([])

  // Modal & Form State
  const [showRequestModal, setShowRequestModal] = useState(false)
  const [reqDeptCode, setReqDeptCode] = useState('')
  const [reqPerms, setReqPerms] = useState<PermissionCode[]>(['VIEW'])
  const [reqReason, setReqReason] = useState('')
  const [submittingReq, setSubmittingReq] = useState(false)

  // Layout & UI State
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false)
  const [activeView, setActiveView] = useState<NavItem>('chat')
  const [toast, setToast] = useState<{ type: 'success' | 'error'; message: string } | null>(null)
  const [conversations, setConversations] = useState<ConversationItem[]>([])
  const [loadingConversations, setLoadingConversations] = useState(false)

  // Workspaces State
  const [workspaces, setWorkspaces] = useState<WorkspaceItem[]>([])
  const [settingsWorkspaceModal, setSettingsWorkspaceModal] = useState<WorkspaceItem | null>(null)
  
  const location = useLocation()
  const navigate = useNavigate()
  
  const workspaceMatch = matchPath('/workspace/:workspaceId', location.pathname.replace('/app', ''))
  const threadMatch = matchPath('/workspace/:workspaceId/thread/:threadId', location.pathname.replace('/app', ''))
  
  const isWorkspaceRoute = !!(workspaceMatch || threadMatch)
  
  const activeWorkspaceId = threadMatch?.params.workspaceId || workspaceMatch?.params.workspaceId || null
  const activeConversationId = threadMatch?.params.threadId || null

  useEffect(() => {
    if (isWorkspaceRoute && activeView !== 'chat') {
      setActiveView('chat')
    }
  }, [isWorkspaceRoute, activeView])

  useEffect(() => {
    if (location.pathname === '/app' || location.pathname === '/app/') {
      if (workspaces.length > 0) {
        navigate(`/app/workspace/${workspaces[0].id}`, { replace: true })
      }
    }
  }, [location.pathname, workspaces, navigate])

  // Documents view state
  const [documents, setDocuments] = useState<DocumentListItem[]>([])
  const [filteredDocuments, setFilteredDocuments] = useState<DocumentListItem[]>([])
  const [docSearch, setDocSearch] = useState('')
  const [docDeptFilter, setDocDeptFilter] = useState('ALL')
  const [docsLoading, setDocsLoading] = useState(false)
  const [previewDoc, setPreviewDoc] = useState<DocumentListItem | null>(null)
  const [previewContent, setPreviewContent] = useState<string | null>(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [previewError, setPreviewError] = useState<string | null>(null)

  const showToast = useCallback((type: 'success' | 'error', message: string) => {
    setToast({ type, message })
    setTimeout(() => setToast(null), 4000)
  }, [])

  /* ---- Load Workspaces ---- */
  const loadWorkspaces = useCallback(async () => {
    if (!token) return
    try {
      const items = await listWorkspaces(token)
      setWorkspaces(items)
    } catch (err) {
      console.error('Failed to load workspaces:', err)
    }
  }, [token, activeWorkspaceId])

  useEffect(() => {
    loadWorkspaces()
  }, [loadWorkspaces])

  /* ---- Load user profile & access ---- */
  const loadUserAndAccess = useCallback(async () => {
    try {
      const user = await userApi.getMe(token)
      if (user.role === 'ADMIN') {
        navigate('/admin', { replace: true })
        return
      }
      setUserEmail(user.email)
      const namePart = user.email.split('@')[0]
      setUserName(namePart.charAt(0).toUpperCase() + namePart.slice(1))

      const myAccess = await userApi.getMyAccess(token)
      setAccess(myAccess)
    } catch {
      onLogout()
    }
  }, [token, navigate, onLogout])

  /* ---- Load Conversations ---- */
  const loadConversations = useCallback(async () => {
    if (!token) return []
    setLoadingConversations(true)
    try {
      const items = await listConversations(token)
      setConversations(items)
      return items
    } catch (err) {
      console.error('Failed to load conversations:', err)
      return []
    } finally {
      setLoadingConversations(false)
    }
  }, [token])

  useEffect(() => {
    // loadConversations side effect removed, driven by URLs now
  }, [loadConversations])

  const handleSelectConversation = (convId: string, workspaceId?: string | null) => {
    const targetWs = workspaceId || activeWorkspaceId
    if (targetWs) {
      navigate(`/app/workspace/${targetWs}/thread/${convId}`)
    }
  }

  const handleSelectWorkspace = (wsId: string) => {
    navigate(`/app/workspace/${wsId}`)
  }

  const handleOpenWorkspaceSettings = (ws: WorkspaceItem) => {
    setSettingsWorkspaceModal(ws)
  }

  const handleNewChat = (workspaceId?: string | null) => {
    const targetWs = workspaceId !== undefined ? workspaceId : activeWorkspaceId
    if (targetWs) {
      navigate(`/app/workspace/${targetWs}`)
    }
  }

  const handleDeleteConversation = async (e: React.MouseEvent, convId: string) => {
    e.stopPropagation()
    try {
      await deleteConversation(convId, token)
      setConversations((prev) => {
        const remaining = prev.filter((c) => c.id !== convId)
        if (activeConversationId === convId) {
          if (activeWorkspaceId) {
            navigate(`/app/workspace/${activeWorkspaceId}`)
          }
        }
        return remaining
      })
    } catch (err) {
      console.error('Failed to delete conversation:', err)
    }
  }


  /* ---- Load My Requests ---- */
  const loadMyRequests = useCallback(async () => {
    try {
      const reqs = await accessRequestApi.getMyAccessRequests(token)
      setMyRequests(reqs)
    } catch (e: any) {
      console.error('Failed to load my access requests', e)
    }
  }, [token])

  /* ---- Load Active Departments ---- */
  const loadActiveDepartments = useCallback(async () => {
    try {
      const depts = await departmentApi.getActiveDepartments(token)
      setActiveDepts(depts)
    } catch (e: any) {
      console.error('Failed to load active departments', e)
    }
  }, [token])

  /* ---- Load Documents ---- */
  const loadDocuments = useCallback(async () => {
    if (!token) return
    setDocsLoading(true)
    try {
      const data = await documentApi.listDocuments(token)
      setDocuments(data)
    } catch (e: any) {
      console.error('Failed to load documents:', e)
      showToast('error', 'Failed to load documents')
    } finally {
      setDocsLoading(false)
    }
  }, [token, showToast])

  useEffect(() => {
    if (activeView === 'sources') loadDocuments()
  }, [activeView, loadDocuments])

  /* ---- Documents Filter ---- */
  useEffect(() => {
    let result = documents
    if (docSearch.trim()) {
      const q = docSearch.toLowerCase()
      result = result.filter(
        (d) =>
          d.file_name.toLowerCase().includes(q) ||
          (d.department_name && d.department_name.toLowerCase().includes(q)) ||
          (d.department_code && d.department_code.toLowerCase().includes(q)) ||
          (d.workspace_name && d.workspace_name.toLowerCase().includes(q)) ||
          (d.workspace_code && d.workspace_code.toLowerCase().includes(q)) ||
          (d.uploaded_by_name && d.uploaded_by_name.toLowerCase().includes(q)) ||
          (d.uploaded_by_email && d.uploaded_by_email.toLowerCase().includes(q))
      )
    }
    if (docDeptFilter !== 'ALL') {
      result = result.filter((d) => d.department_code === docDeptFilter || d.department_id === docDeptFilter)
    }
    setFilteredDocuments(result)
  }, [documents, docSearch, docDeptFilter])

  /* ---- Preview Document Handler ---- */
  const handleOpenPreview = async (doc: DocumentListItem) => {
    setPreviewDoc(doc)
    setPreviewContent(null)
    setPreviewError(null)
    setPreviewLoading(true)
    try {
      const res = await fetch(`/api/documents/${encodeURIComponent(doc.id)}/preview`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      if (!res.ok) {
        throw new Error(`Failed to load preview (${res.status} ${res.statusText})`)
      }
      const text = await res.text()
      setPreviewContent(text)
    } catch (err: any) {
      setPreviewError(err.message || 'Failed to load document preview')
    } finally {
      setPreviewLoading(false)
    }
  }

  useEffect(() => {
    loadUserAndAccess()
    loadMyRequests()
    loadActiveDepartments()
  }, [loadUserAndAccess, loadMyRequests, loadActiveDepartments])

  /* ---- Submit Access Request ---- */
  async function handleSubmitRequest() {
    const code = reqDeptCode.trim().toUpperCase()
    if (!code) {
      showToast('error', 'Please select a department.')
      return
    }
    if (reqPerms.length === 0) {
      showToast('error', 'Please select at least one permission.')
      return
    }

    setSubmittingReq(true)
    try {
      await accessRequestApi.createAccessRequest(
        {
          department_code: code,
          requested_permissions: reqPerms,
          reason: reqReason.trim() || undefined,
        },
        token,
      )
      showToast('success', `Access request for ${code} submitted successfully.`)
      setShowRequestModal(false)
      setReqDeptCode('')
      setReqPerms(['VIEW'])
      setReqReason('')
      await Promise.all([loadMyRequests(), loadUserAndAccess()])
    } catch (e: any) {
      showToast('error', e.message)
    } finally {
      setSubmittingReq(false)
    }
  }

  function handleTogglePerm(perm: PermissionCode) {
    if (reqPerms.includes(perm)) {
      setReqPerms(reqPerms.filter((p) => p !== perm))
    } else {
      setReqPerms([...reqPerms, perm])
    }
  }

  if (!access) {
    return (
      <div
        style={{
          minHeight: '100dvh',
          background: 'var(--color-ground)',
          color: 'var(--color-ink-muted)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontFamily: 'var(--font-sans)',
        }}
      >
        Loading access profile…
      </div>
    )
  }

  const allowedUploadDepts = activeDepts
    .filter(
      (d) =>
        d.is_active &&
        access[d.code] &&
        (access[d.code].includes('UPLOAD') || access[d.code].includes('ALL')),
    )
    .map((d) => ({ code: d.code, name: d.name }))

  const hasNoAccess = Object.keys(access).length === 0

  if (activeView === 'chat') {
    return (
      <AnythingLLMApp
        token={token}
        activeWorkspaceId={activeWorkspaceId}
        activeThreadId={activeConversationId}
        onSelectWorkspace={handleSelectWorkspace}
        onSelectThread={handleSelectConversation}
        onNewThread={handleNewChat}
        onLogout={onLogout}
      />
    )
  }

  return (
    <div
      style={{
        display: 'flex',
        height: '100vh',
        maxHeight: '100vh',
        width: '100vw',
        overflow: 'hidden',
        background: 'var(--color-ground)',
        fontFamily: 'var(--font-sans)',
        color: 'var(--color-ink)',
      }}
    >
      <Sidebar
        collapsed={isSidebarCollapsed}
        onToggleCollapse={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
        role="USER"
        userName={userName}
        userEmail={userEmail}
        onLogout={onLogout}
        activeItem={activeView}
        onSelect={(item) => setActiveView(item)}
      />

      {/* WorkspacePanel only visible on chat view */}
      {(activeView as string) === 'chat' && (
        <WorkspacePanel
          workspaces={workspaces}
          activeWorkspaceId={activeWorkspaceId}
          onSelectWorkspace={handleSelectWorkspace}
          onOpenWorkspaceSettings={handleOpenWorkspaceSettings}
          conversations={conversations}
          activeConversationId={activeConversationId}
          onSelectConversation={handleSelectConversation}
          onNewChat={handleNewChat}
          onDeleteConversation={handleDeleteConversation}
          loadingConversations={loadingConversations}
        />
      )}

      <div style={{ flex: 1, minWidth: 0, height: '100vh', maxHeight: '100vh', display: 'flex', flexDirection: 'column', position: 'relative', overflow: 'hidden' }}>
        {/* Toast */}
        {toast && (
          <div
            style={{
              position: 'fixed',
              top: '20px',
              right: '24px',
              zIndex: 4000,
              padding: '12px 20px',
              borderRadius: '10px',
              border: '1px solid',
              fontSize: '13px',
              fontWeight: 500,
              boxShadow: '0 8px 24px rgba(0,0,0,0.12)',
              background: toast.type === 'error' ? '#fef2f2' : '#f0fdf4',
              borderColor: toast.type === 'error' ? '#fecaca' : '#bbf7d0',
              color: toast.type === 'error' ? '#991b1b' : '#166534',
              animation: 'fadeIn 0.2s ease',
            }}
          >
            {toast.message}
          </div>
        )}

        {/* Modal for Workspace Provider Profile Settings */}
        {settingsWorkspaceModal && (
          <WorkspaceProviderSettingsModal
            workspace={settingsWorkspaceModal}
            token={token}
            isOwnerOrAdmin={true}
            onClose={() => setSettingsWorkspaceModal(null)}
            onToast={showToast}
          />
        )}

        {/* VIEW 1: Dashboard / Chat */}
        {(activeView as string) === 'chat' && (
          <RAG
            token={token}
            userName={userName}
            allowedUploadDepts={allowedUploadDepts}
            activeConversationId={activeConversationId}
            activeConversationTitle={conversations.find((c) => c.id === activeConversationId)?.title}
            onConversationCreatedOrUpdated={(convId) => {
              loadConversations()
            }}
            onToggleSidebar={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
            isSidebarCollapsed={isSidebarCollapsed}
            workspaces={workspaces}
            activeWorkspaceId={activeWorkspaceId}
            onSelectWorkspace={handleSelectWorkspace}
            onOpenWorkspaceSettings={handleOpenWorkspaceSettings}
            onNavigateView={(view) => setActiveView(view)}
          />
        )}

        {/* VIEW: Documents */}
        {activeView === 'sources' && (
          <div style={{ flex: 1, minHeight: 0, height: '100%', padding: '48px 32px', overflowY: 'auto', overflowX: 'hidden' }}>
            <div style={{ maxWidth: '1000px', margin: '0 auto' }}>
              {/* Header */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '28px', gap: '16px' }}>
                <div>
                  <h1 style={{ fontFamily: 'var(--font-serif)', fontSize: '28px', margin: '0 0 6px 0', color: 'var(--color-ink)' }}>
                    Documents
                  </h1>
                  <p style={{ margin: 0, color: 'var(--color-ink-muted)', fontSize: '14px' }}>
                    {documents.length} document{documents.length !== 1 ? 's' : ''} available based on your department permissions
                  </p>
                </div>
              </div>

              {/* Filters */}
              <div style={{ display: 'flex', gap: '12px', marginBottom: '20px', flexWrap: 'wrap' }}>
                <div style={{ flex: 1, minWidth: '220px', position: 'relative' }}>
                  <input
                    placeholder="Search by file name or uploader…"
                    value={docSearch}
                    onChange={(e) => setDocSearch(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '10px 14px',
                      borderRadius: '8px',
                      border: '1px solid var(--color-border)',
                      background: 'var(--color-surface)',
                      fontSize: '13px',
                      color: 'var(--color-ink)',
                      boxSizing: 'border-box',
                    }}
                  />
                </div>

                <select
                  value={docDeptFilter}
                  onChange={(e) => setDocDeptFilter(e.target.value)}
                  style={{
                    padding: '10px 14px',
                    borderRadius: '8px',
                    border: '1px solid var(--color-border)',
                    background: 'var(--color-surface)',
                    fontSize: '13px',
                    color: 'var(--color-ink)',
                  }}
                >
                  <option value="ALL">All Authorized Departments</option>
                  {activeDepts.map((d) => (
                    <option key={d.code} value={d.code}>
                      {d.name} ({d.code})
                    </option>
                  ))}
                </select>
              </div>

              {/* Documents Table Card */}
              <div
                style={{
                  background: 'var(--color-surface)',
                  borderRadius: '12px',
                  border: '1px solid var(--color-border)',
                  overflowX: 'auto',
                  width: '100%',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.02)',
                }}
              >
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '13px' }}>
                  <thead>
                    <tr style={{ background: 'var(--color-panel)', borderBottom: '1px solid var(--color-border)' }}>
                      <th style={{ padding: '12px 16px', color: 'var(--color-ink-muted)', fontWeight: 600, fontFamily: 'var(--font-mono)', fontSize: '11px', textTransform: 'uppercase' }}>File Name</th>
                      <th style={{ padding: '12px 16px', color: 'var(--color-ink-muted)', fontWeight: 600, fontFamily: 'var(--font-mono)', fontSize: '11px', textTransform: 'uppercase' }}>Department</th>
                      <th style={{ padding: '12px 16px', color: 'var(--color-ink-muted)', fontWeight: 600, fontFamily: 'var(--font-mono)', fontSize: '11px', textTransform: 'uppercase' }}>Workspace</th>
                      <th style={{ padding: '12px 16px', color: 'var(--color-ink-muted)', fontWeight: 600, fontFamily: 'var(--font-mono)', fontSize: '11px', textTransform: 'uppercase' }}>Uploaded By</th>
                      <th style={{ padding: '12px 16px', color: 'var(--color-ink-muted)', fontWeight: 600, fontFamily: 'var(--font-mono)', fontSize: '11px', textTransform: 'uppercase' }}>Uploaded At</th>
                      <th style={{ padding: '12px 16px', color: 'var(--color-ink-muted)', fontWeight: 600, fontFamily: 'var(--font-mono)', fontSize: '11px', textTransform: 'uppercase', textAlign: 'right' }}>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {docsLoading ? (
                      <tr>
                        <td colSpan={6} style={{ padding: '32px', textAlign: 'center', color: 'var(--color-ink-muted)' }}>
                          Loading documents…
                        </td>
                      </tr>
                    ) : filteredDocuments.length === 0 ? (
                      <tr>
                        <td colSpan={6} style={{ padding: '32px', textAlign: 'center', color: 'var(--color-ink-muted)' }}>
                          No documents found.
                        </td>
                      </tr>
                    ) : (
                      filteredDocuments.map((doc) => (
                        <tr key={doc.id} style={{ borderBottom: '1px solid var(--color-border)' }}>
                          <td style={{ padding: '14px 16px' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                              <span style={{ fontSize: '16px' }}>📄</span>
                              <div>
                                <div style={{ fontWeight: 600, color: 'var(--color-ink)' }}>{doc.file_name}</div>
                                <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)', fontFamily: 'var(--font-mono)' }}>
                                  {doc.mime_type || 'Document'}
                                </div>
                              </div>
                            </div>
                          </td>
                          <td style={{ padding: '14px 16px' }}>
                            <span
                              style={{
                                display: 'inline-block',
                                padding: '3px 9px',
                                borderRadius: '12px',
                                background: 'var(--color-teal-light)',
                                color: 'var(--color-teal)',
                                border: '1px solid var(--color-teal)',
                                fontSize: '12px',
                                fontWeight: 500,
                              }}
                            >
                              {doc.department_name || doc.department_code || 'Unassigned'}
                            </span>
                          </td>
                          <td style={{ padding: '14px 16px' }}>
                            {doc.workspace_name || doc.workspace_code ? (
                              <span
                                style={{
                                  display: 'inline-block',
                                  padding: '3px 9px',
                                  borderRadius: '12px',
                                  background: 'var(--color-surface)',
                                  color: 'var(--color-ink)',
                                  border: '1px solid var(--color-border)',
                                  fontSize: '12px',
                                  fontWeight: 500,
                                }}
                              >
                                {doc.workspace_name ? `${doc.workspace_name} (${doc.workspace_code})` : doc.workspace_code}
                              </span>
                            ) : (
                              <span style={{ fontSize: '12px', color: 'var(--color-ink-muted)' }}>—</span>
                            )}
                          </td>
                          <td style={{ padding: '14px 16px' }}>
                            <div style={{ fontSize: '13px', color: 'var(--color-ink)', fontWeight: 500 }}>
                              {doc.uploaded_by_name || (doc.uploaded_by_email ? doc.uploaded_by_email.split('@')[0] : 'System')}
                            </div>
                            {doc.uploaded_by_email && (
                              <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)', fontFamily: 'var(--font-mono)' }}>
                                {doc.uploaded_by_email}
                              </div>
                            )}
                          </td>
                          <td style={{ padding: '14px 16px' }}>
                            <div style={{ fontSize: '12px', color: 'var(--color-ink)' }}>
                              {new Date(doc.created_at).toLocaleDateString()}
                            </div>
                            <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)' }}>
                              {new Date(doc.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                            </div>
                          </td>
                          <td style={{ padding: '14px 16px', textAlign: 'right' }}>
                            <button
                              onClick={() => handleOpenPreview(doc)}
                              style={{
                                padding: '6px 14px',
                                borderRadius: '6px',
                                background: 'var(--color-panel)',
                                border: '1px solid var(--color-border)',
                                color: 'var(--color-ink)',
                                cursor: 'pointer',
                                fontSize: '12px',
                                fontWeight: 500,
                              }}
                            >
                              👁 Preview
                            </button>
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}


        {/* VIEW 2: My Access */}
        {activeView === 'access' && (
          <div style={{ flex: 1, minHeight: 0, height: '100%', padding: '48px 32px', overflowY: 'auto', overflowX: 'hidden' }}>
            <div style={{ maxWidth: '800px', margin: '0 auto' }}>
              {/* Header */}
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'flex-start',
                  marginBottom: '28px',
                  gap: '16px',
                }}
              >
                <div>
                  <h1
                    style={{
                      fontFamily: 'var(--font-serif)',
                      fontSize: '28px',
                      margin: '0 0 6px 0',
                    }}
                  >
                    My Access & Permissions
                  </h1>
                  <p
                    style={{
                      fontSize: '14px',
                      color: 'var(--color-ink-muted)',
                      margin: 0,
                    }}
                  >
                    Review granted department permissions and submit access requests.
                  </p>
                </div>

                <button
                  onClick={() => {
                    if (activeDepts.length > 0 && !reqDeptCode) {
                      setReqDeptCode(activeDepts[0].code)
                    }
                    setShowRequestModal(true)
                  }}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    padding: '10px 18px',
                    borderRadius: '8px',
                    background: 'var(--color-teal)',
                    color: 'white',
                    border: 'none',
                    fontWeight: 600,
                    fontSize: '13px',
                    cursor: 'pointer',
                    flexShrink: 0,
                    boxShadow: '0 2px 8px rgba(0, 105, 92, 0.2)',
                  }}
                >
                  <svg
                    width="16"
                    height="16"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                  >
                    <line x1="12" y1="5" x2="12" y2="19" />
                    <line x1="5" y1="12" x2="19" y2="12" />
                  </svg>
                  Request Access
                </button>
              </div>

              {/* Current Active Access Cards */}
              <div style={{ marginBottom: '40px' }}>
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    marginBottom: '16px',
                  }}
                >
                  <h2
                    style={{
                      fontSize: '16px',
                      fontWeight: 600,
                      color: 'var(--color-ink)',
                      margin: 0,
                    }}
                  >
                    Current Granted Access
                  </h2>
                  <span
                    style={{
                      fontSize: '12px',
                      fontFamily: 'var(--font-mono)',
                      color: hasNoAccess ? '#b45309' : '#047857',
                      background: hasNoAccess ? '#fff8e1' : '#ecfdf5',
                      border: `1px solid ${hasNoAccess ? '#fde68a' : '#a7f3d0'}`,
                      padding: '2px 8px',
                      borderRadius: '12px',
                      fontWeight: 600,
                    }}
                  >
                    {hasNoAccess ? 'No Access' : `${Object.keys(access).length} Departments`}
                  </span>
                </div>

                {hasNoAccess ? (
                  <div
                    style={{
                      background: 'var(--color-surface)',
                      border: '1px solid var(--color-border)',
                      padding: '32px 24px',
                      borderRadius: '12px',
                      textAlign: 'center',
                    }}
                  >
                    <div style={{ fontSize: '32px', marginBottom: '8px' }}>🔒</div>
                    <div
                      style={{
                        fontWeight: 600,
                        fontSize: '16px',
                        color: 'var(--color-ink)',
                        marginBottom: '4px',
                      }}
                    >
                      No Department Access Assigned
                    </div>
                    <p
                      style={{
                        fontSize: '13px',
                        color: 'var(--color-ink-muted)',
                        maxWidth: '480px',
                        margin: '0 auto 20px',
                        lineHeight: '1.5',
                      }}
                    >
                      Your account currently does not have access to any department documents or query capabilities. Click <strong>Request Access</strong> to submit a request for an active department.
                    </p>
                    <button
                      onClick={() => {
                        if (activeDepts.length > 0 && !reqDeptCode) {
                          setReqDeptCode(activeDepts[0].code)
                        }
                        setShowRequestModal(true)
                      }}
                      style={{
                        padding: '8px 20px',
                        borderRadius: '8px',
                        background: 'var(--color-teal)',
                        color: 'white',
                        border: 'none',
                        fontSize: '13px',
                        fontWeight: 600,
                        cursor: 'pointer',
                      }}
                    >
                      Request Department Access
                    </button>
                  </div>
                ) : (
                  <div style={{ display: 'grid', gap: '16px' }}>
                    {Object.entries(access).map(([deptCode, perms]) => {
                      const deptObj = activeDepts.find((d) => d.code === deptCode)
                      const displayName = deptObj?.name || deptCode
                      return (
                        <div
                          key={deptCode}
                          style={{
                            background: 'var(--color-surface)',
                            border: '1px solid var(--color-border)',
                            borderRadius: '12px',
                            padding: '20px 24px',
                          }}
                        >
                          <div
                            style={{
                              display: 'flex',
                              justifyContent: 'space-between',
                              alignItems: 'center',
                              marginBottom: '14px',
                            }}
                          >
                            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                              <div
                                style={{
                                  fontSize: '15px',
                                  fontWeight: 600,
                                  color: 'var(--color-ink)',
                                }}
                              >
                                {displayName}
                              </div>
                            </div>
                            <span
                              style={{
                                fontSize: '12px',
                                color: '#047857',
                                fontWeight: 500,
                                background: '#ecfdf5',
                                padding: '3px 10px',
                                borderRadius: '12px',
                                border: '1px solid #a7f3d0',
                              }}
                            >
                              Active Access
                            </span>
                          </div>

                        <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
                          {ALL_PERMS.map((p) => {
                            const hasPerm = perms.includes(p) || perms.includes('ALL')
                            return (
                              <div
                                key={p}
                                style={{
                                  display: 'flex',
                                  alignItems: 'center',
                                  gap: '8px',
                                  opacity: hasPerm ? 1 : 0.35,
                                }}
                              >
                                <div
                                  style={{
                                    width: '18px',
                                    height: '18px',
                                    borderRadius: '4px',
                                    border: hasPerm
                                      ? 'none'
                                      : '1px solid var(--color-border)',
                                    background: hasPerm
                                      ? 'var(--color-teal)'
                                      : 'transparent',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                  }}
                                >
                                  {hasPerm && (
                                    <svg
                                      width="12"
                                      height="12"
                                      viewBox="0 0 24 24"
                                      fill="none"
                                      stroke="white"
                                      strokeWidth="3"
                                      strokeLinecap="round"
                                      strokeLinejoin="round"
                                    >
                                      <polyline points="20 6 9 17 4 12"></polyline>
                                    </svg>
                                  )}
                                </div>
                                <span
                                  style={{
                                    fontSize: '12px',
                                    fontFamily: 'var(--font-mono)',
                                    fontWeight: 600,
                                    color: 'var(--color-ink)',
                                  }}
                                >
                                  {p}
                                </span>
                              </div>
                            )
                          })}
                        </div>
                      </div>
                    )
                  })}
                </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* VIEW 3: My Requests */}
        {activeView === 'my-requests' && (
          <div style={{ flex: 1, minHeight: 0, height: '100%', padding: '48px 32px', overflowY: 'auto', overflowX: 'hidden' }}>
            <div style={{ maxWidth: '800px', margin: '0 auto' }}>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'flex-start',
                  marginBottom: '28px',
                  gap: '16px',
                }}
              >
                <div>
                  <h1
                    style={{
                      fontFamily: 'var(--font-serif)',
                      fontSize: '28px',
                      margin: '0 0 6px 0',
                    }}
                  >
                    My Access Requests
                  </h1>
                  <p
                    style={{
                      fontSize: '14px',
                      color: 'var(--color-ink-muted)',
                      margin: 0,
                    }}
                  >
                    Track status and administrator reviews for all your submitted access requests.
                  </p>
                </div>

                <button
                  onClick={() => {
                    if (activeDepts.length > 0 && !reqDeptCode) {
                      setReqDeptCode(activeDepts[0].code)
                    }
                    setShowRequestModal(true)
                  }}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    padding: '10px 18px',
                    borderRadius: '8px',
                    background: 'var(--color-teal)',
                    color: 'white',
                    border: 'none',
                    fontWeight: 600,
                    fontSize: '13px',
                    cursor: 'pointer',
                    flexShrink: 0,
                    boxShadow: '0 2px 8px rgba(0, 105, 92, 0.2)',
                  }}
                >
                  <svg
                    width="16"
                    height="16"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                  >
                    <line x1="12" y1="5" x2="12" y2="19" />
                    <line x1="5" y1="12" x2="19" y2="12" />
                  </svg>
                  New Request
                </button>
              </div>

              {myRequests.length === 0 ? (
                <div
                  style={{
                    background: 'var(--color-surface)',
                    border: '1px solid var(--color-border)',
                    padding: '40px 24px',
                    borderRadius: '12px',
                    textAlign: 'center',
                    color: 'var(--color-ink-muted)',
                  }}
                >
                  <div style={{ fontSize: '32px', marginBottom: '8px' }}>📋</div>
                  <div style={{ fontWeight: 600, fontSize: '15px', color: 'var(--color-ink)', marginBottom: '4px' }}>
                    No Access Requests Found
                  </div>
                  <p style={{ fontSize: '13px', margin: '0 0 16px 0' }}>
                    You have not submitted any access requests yet.
                  </p>
                  <button
                    onClick={() => {
                      if (activeDepts.length > 0 && !reqDeptCode) {
                        setReqDeptCode(activeDepts[0].code)
                      }
                      setShowRequestModal(true)
                    }}
                    style={{
                      padding: '8px 18px',
                      borderRadius: '8px',
                      background: 'var(--color-teal)',
                      color: 'white',
                      border: 'none',
                      fontSize: '13px',
                      fontWeight: 600,
                      cursor: 'pointer',
                    }}
                  >
                    Submit First Request
                  </button>
                </div>
              ) : (
                <div
                  style={{
                    background: 'var(--color-surface)',
                    border: '1px solid var(--color-border)',
                    borderRadius: '12px',
                    overflowX: 'auto',
                    width: '100%',
                  }}
                >
                  <table
                    style={{
                      width: '100%',
                      borderCollapse: 'collapse',
                      textAlign: 'left',
                      fontSize: '13px',
                    }}
                  >
                    <thead>
                      <tr
                        style={{
                          background: 'var(--color-ground)',
                          borderBottom: '1px solid var(--color-border)',
                        }}
                      >
                        <th style={{ padding: '12px 16px', fontWeight: 600 }}>Department</th>
                        <th style={{ padding: '12px 16px', fontWeight: 600 }}>Requested Permissions</th>
                        <th style={{ padding: '12px 16px', fontWeight: 600 }}>Status</th>
                        <th style={{ padding: '12px 16px', fontWeight: 600 }}>Requested At</th>
                        <th style={{ padding: '12px 16px', fontWeight: 600 }}>Review Details</th>
                      </tr>
                    </thead>
                    <tbody>
                      {myRequests.map((req) => {
                        const badge = statusBadge(req.status)
                        return (
                          <tr
                            key={req.id}
                            style={{
                              borderBottom: '1px solid var(--color-border)',
                            }}
                          >
                            <td style={{ padding: '14px 16px' }}>
                              <div
                                style={{
                                  fontWeight: 600,
                                  fontSize: '14px',
                                  color: 'var(--color-ink)',
                                }}
                              >
                                {req.department_name}
                              </div>
                            </td>
                            <td style={{ padding: '14px 16px' }}>
                              <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
                                {req.requested_permissions.map((p) => (
                                  <span
                                    key={p}
                                    style={{
                                      fontSize: '11px',
                                      fontFamily: 'var(--font-mono)',
                                      padding: '2px 6px',
                                      borderRadius: '4px',
                                      background: '#e0f2fe',
                                      color: '#0369a1',
                                      border: '1px solid #bae6fd',
                                      fontWeight: 600,
                                    }}
                                  >
                                    {p}
                                  </span>
                                ))}
                              </div>
                            </td>
                            <td style={{ padding: '14px 16px' }}>
                              <span
                                style={{
                                  display: 'inline-block',
                                  padding: '3px 9px',
                                  borderRadius: '12px',
                                  background: badge.bg,
                                  color: badge.color,
                                  border: `1px solid ${badge.border}`,
                                  fontSize: '12px',
                                  fontWeight: 600,
                                }}
                              >
                                {badge.label}
                              </span>
                            </td>
                            <td style={{ padding: '14px 16px' }}>
                              <div style={{ fontSize: '12px', color: 'var(--color-ink)' }}>
                                {new Date(req.requested_at).toLocaleDateString()}
                              </div>
                              <div
                                style={{
                                  fontSize: '11px',
                                  color: 'var(--color-ink-muted)',
                                }}
                              >
                                {new Date(req.requested_at).toLocaleTimeString([], {
                                  hour: '2-digit',
                                  minute: '2-digit',
                                })}
                              </div>
                            </td>
                            <td style={{ padding: '14px 16px' }}>
                              {req.reviewed_at ? (
                                <div>
                                  <div
                                    style={{
                                      fontSize: '12px',
                                      color: 'var(--color-ink)',
                                      fontWeight: 500,
                                    }}
                                  >
                                    Reviewed {new Date(req.reviewed_at).toLocaleDateString()}
                                  </div>
                                  {req.review_note && (
                                    <div
                                      style={{
                                        fontSize: '11px',
                                        color: 'var(--color-ink-muted)',
                                        fontStyle: 'italic',
                                        marginTop: '2px',
                                      }}
                                    >
                                      "{req.review_note}"
                                    </div>
                                  )}
                                </div>
                              ) : (
                                <span
                                  style={{
                                    fontSize: '12px',
                                    color: 'var(--color-ink-muted)',
                                  }}
                                >
                                  Awaiting Admin review
                                </span>
                              )}
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* ====== REQUEST ACCESS MODAL ====== */}
      {showRequestModal && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(21, 29, 50, 0.5)',
            backdropFilter: 'blur(4px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 3000,
          }}
          onClick={() => setShowRequestModal(false)}
        >
          <div
            style={{
              background: 'var(--color-surface)',
              borderRadius: '16px',
              width: '100%',
              maxWidth: '480px',
              boxShadow: '0 20px 60px rgba(0,0,0,0.15)',
              overflow: 'hidden',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '20px 24px',
                borderBottom: '1px solid var(--color-border)',
              }}
            >
              <h2
                style={{
                  margin: 0,
                  fontFamily: 'var(--font-serif)',
                  fontSize: '20px',
                }}
              >
                Request Department Access
              </h2>
              <button
                onClick={() => setShowRequestModal(false)}
                style={{
                  background: 'transparent',
                  border: 'none',
                  cursor: 'pointer',
                  color: 'var(--color-ink-muted)',
                  padding: '4px',
                }}
              >
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                >
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>

            <div
              style={{
                padding: '24px',
                display: 'flex',
                flexDirection: 'column',
                gap: '16px',
              }}
            >
              {/* Department Dropdown */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                <label
                  style={{
                    fontSize: '12px',
                    fontFamily: 'var(--font-mono)',
                    fontWeight: 600,
                    color: 'var(--color-ink-muted)',
                    textTransform: 'uppercase',
                    letterSpacing: '0.06em',
                  }}
                >
                  Select Department
                </label>
                <select
                  value={reqDeptCode}
                  onChange={(e) => setReqDeptCode(e.target.value)}
                  style={{
                    padding: '10px 14px',
                    borderRadius: '10px',
                    border: '1px solid var(--color-border)',
                    background: 'var(--color-ground)',
                    fontSize: '14px',
                    color: 'var(--color-ink)',
                    fontFamily: 'var(--font-sans)',
                  }}
                >
                  {activeDepts.length === 0 ? (
                    <option value="">No active departments found</option>
                  ) : (
                    activeDepts.map((d) => (
                      <option key={d.code} value={d.code}>
                        {d.name}
                      </option>
                    ))
                  )}
                </select>
              </div>

              {/* Requested Permissions */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                  }}
                >
                  <label
                    style={{
                      fontSize: '12px',
                      fontFamily: 'var(--font-mono)',
                      fontWeight: 600,
                      color: 'var(--color-ink-muted)',
                      textTransform: 'uppercase',
                      letterSpacing: '0.06em',
                    }}
                  >
                    Requested Permissions
                  </label>
                  <div style={{ display: 'flex', gap: '6px' }}>
                    {Object.entries(PRESETS).map(([name, codes]) => (
                      <button
                        key={name}
                        type="button"
                        onClick={() => setReqPerms([...codes])}
                        style={{
                          padding: '2px 8px',
                          fontSize: '11px',
                          borderRadius: '4px',
                          border: '1px solid var(--color-border)',
                          background: 'var(--color-surface)',
                          color: 'var(--color-ink-muted)',
                          cursor: 'pointer',
                        }}
                      >
                        {name}
                      </button>
                    ))}
                  </div>
                </div>

                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(2, 1fr)',
                    gap: '8px',
                    marginTop: '4px',
                  }}
                >
                  {ALL_PERMS.map((p) => {
                    const checked = reqPerms.includes(p)
                    return (
                      <label
                        key={p}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '8px',
                          padding: '10px 12px',
                          borderRadius: '8px',
                          border: `1px solid ${
                            checked ? 'var(--color-teal)' : 'var(--color-border)'
                          }`,
                          background: checked
                            ? 'rgba(0, 105, 92, 0.04)'
                            : 'var(--color-surface)',
                          cursor: 'pointer',
                        }}
                      >
                        <input
                          type="checkbox"
                          checked={checked}
                          onChange={() => handleTogglePerm(p)}
                          style={{ accentColor: 'var(--color-teal)' }}
                        />
                        <span
                          style={{
                            fontWeight: 600,
                            fontSize: '13px',
                            fontFamily: 'var(--font-mono)',
                            color: 'var(--color-ink)',
                          }}
                        >
                          {p}
                        </span>
                      </label>
                    )
                  })}
                </div>
              </div>

              {/* Optional Reason */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                <label
                  style={{
                    fontSize: '12px',
                    fontFamily: 'var(--font-mono)',
                    fontWeight: 600,
                    color: 'var(--color-ink-muted)',
                    textTransform: 'uppercase',
                    letterSpacing: '0.06em',
                  }}
                >
                  Reason / Justification (Optional)
                </label>
                <textarea
                  rows={2}
                  placeholder="Explain why you need access to this department…"
                  value={reqReason}
                  onChange={(e) => setReqReason(e.target.value)}
                  style={{
                    padding: '10px 14px',
                    borderRadius: '10px',
                    border: '1px solid var(--color-border)',
                    background: 'var(--color-ground)',
                    fontSize: '13px',
                    color: 'var(--color-ink)',
                    fontFamily: 'var(--font-sans)',
                    resize: 'vertical',
                  }}
                />
              </div>

              {/* Informational Alert */}
              <div
                style={{
                  padding: '12px 14px',
                  borderRadius: '8px',
                  background: '#f0fdf4',
                  border: '1px solid #bbf7d0',
                  fontSize: '12px',
                  lineHeight: '1.4',
                  color: '#166534',
                }}
              >
                <strong>ℹ Submission Notice</strong>
                <div style={{ marginTop: '2px' }}>
                  Submitting a request does not immediately grant access. An administrator will review and authorize your request.
                </div>
              </div>
            </div>

            <div
              style={{
                display: 'flex',
                justifyContent: 'flex-end',
                gap: '8px',
                padding: '16px 24px',
                borderTop: '1px solid var(--color-border)',
              }}
            >
              <button
                onClick={() => setShowRequestModal(false)}
                disabled={submittingReq}
                style={{
                  padding: '8px 20px',
                  borderRadius: '8px',
                  border: '1px solid var(--color-border)',
                  background: 'var(--color-ground)',
                  fontSize: '13px',
                  cursor: 'pointer',
                  color: 'var(--color-ink-muted)',
                }}
              >
                Cancel
              </button>
              <button
                onClick={handleSubmitRequest}
                disabled={!reqDeptCode || reqPerms.length === 0 || submittingReq}
                style={{
                  padding: '8px 24px',
                  borderRadius: '8px',
                  border: 'none',
                  background: 'var(--color-teal)',
                  color: 'white',
                  fontSize: '13px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  opacity: !reqDeptCode || reqPerms.length === 0 ? 0.5 : 1,
                }}
              >
                {submittingReq ? 'Submitting…' : 'Submit Request'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ====== PREVIEW DOCUMENT MODAL ====== */}
      {previewDoc && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(21, 29, 50, 0.5)',
            backdropFilter: 'blur(4px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 3000,
          }}
          onClick={() => setPreviewDoc(null)}
        >
          <div
            style={{
              background: 'var(--color-surface)',
              borderRadius: '16px',
              width: '90%',
              maxWidth: '800px',
              maxHeight: '85vh',
              display: 'flex',
              flexDirection: 'column',
              boxShadow: '0 20px 60px rgba(0,0,0,0.15)',
              overflow: 'hidden',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '20px 24px',
                borderBottom: '1px solid var(--color-border)',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '20px' }}>📄</span>
                <div>
                  <h2 style={{ margin: 0, fontFamily: 'var(--font-serif)', fontSize: '18px', color: 'var(--color-ink)' }}>
                    {previewDoc.file_name}
                  </h2>
                  <div style={{ fontSize: '12px', color: 'var(--color-ink-muted)', marginTop: '2px' }}>
                    Department: <strong>{previewDoc.department_name || previewDoc.department_code || 'Unassigned'}</strong> · Uploaded by:{' '}
                    <strong>{previewDoc.uploaded_by_name || previewDoc.uploaded_by_email || 'System'}</strong>
                  </div>
                </div>
              </div>
              <button
                onClick={() => setPreviewDoc(null)}
                style={{
                  background: 'transparent',
                  border: 'none',
                  cursor: 'pointer',
                  color: 'var(--color-ink-muted)',
                  padding: '4px',
                  borderRadius: '6px',
                }}
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>

            <div style={{ padding: '24px', flex: 1, overflowY: 'auto', minHeight: '280px' }}>
              {previewLoading ? (
                <div style={{ textAlign: 'center', padding: '48px', color: 'var(--color-ink-muted)' }}>
                  Loading document preview…
                </div>
              ) : previewError ? (
                <div style={{ textAlign: 'center', padding: '32px', color: '#b91c1c' }}>
                  {previewError}
                </div>
              ) : previewContent ? (
                <div
                  style={{
                    background: 'var(--color-ground)',
                    border: '1px solid var(--color-border)',
                    borderRadius: '8px',
                    padding: '16px',
                    fontSize: '13px',
                    lineHeight: 1.6,
                    fontFamily: 'var(--font-mono)',
                    whiteSpace: 'pre-wrap',
                    maxHeight: '480px',
                    overflowY: 'auto',
                    color: 'var(--color-ink)',
                  }}
                >
                  {previewContent}
                </div>
              ) : (
                <div style={{ textAlign: 'center', padding: '32px', color: 'var(--color-ink-muted)' }}>
                  No content available for preview.
                </div>
              )}
            </div>

            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '12px 24px',
                borderTop: '1px solid var(--color-border)',
              }}
            >
              <span style={{ fontSize: '11px', color: 'var(--color-ink-muted)', fontFamily: 'var(--font-mono)' }}>
                Read-only preview
              </span>
              <button
                onClick={() => setPreviewDoc(null)}
                style={{
                  padding: '6px 16px',
                  borderRadius: '6px',
                  background: 'var(--color-panel)',
                  border: '1px solid var(--color-border)',
                  color: 'var(--color-ink)',
                  cursor: 'pointer',
                  fontSize: '13px',
                }}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
