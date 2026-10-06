import React, { useState, useEffect, useRef, useCallback } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import {
  WorkspaceItem,
  listWorkspaces,
  createWorkspace,
  WorkspaceMemberItem,
  listWorkspaceMembers,
  addWorkspaceMember,
  removeWorkspaceMember,
  WorkspaceRole,
} from '../api/workspaces'
import {
  DepartmentResponse,
  getActiveDepartments,
  getAllDepartments,
  createDepartment,
  updateDepartmentStatus,
} from '../api/departments'
import { userApi } from '../api/user'
import {
  ConversationItem,
  listConversations,
  createConversation,
  deleteConversation,
  askQuestion,
  getConversation,
  ChatMessageRecord,
} from '../api/chat'
import {
  documentApi,
  DocumentListItem,
  uploadDocument,
} from '../api/documents'
import {
  accessRequestApi,
  AccessRequestResponse,
  PermissionCode,
} from '../api/accessRequests'
import WorkspaceProviderSettingsModal from './WorkspaceProviderSettingsModal'
import WorkspaceMembersModal from './WorkspaceMembersModal'
import { VisualizationRenderer, VisualizationSpec } from './visualizations'
import { MarkdownRenderer } from './MarkdownRenderer'

type GlobalNavView = 'chat' | 'documents' | 'knowledge' | 'admin' | 'departments' | 'requests'
type WorkspaceTab = 'threads' | 'documents' | 'knowledge' | 'members' | 'provider'

interface AnythingLLMAppProps {
  token: string
  activeWorkspaceId?: string | null
  activeThreadId?: string | null
  onSelectWorkspace?: (wsId: string) => void
  onSelectThread?: (threadId: string, wsId: string) => void
  onNewThread?: (wsId?: string) => void
  onLogout?: () => void
}

export default function AnythingLLMApp({
  token,
  activeWorkspaceId: propActiveWorkspaceId,
  activeThreadId: propActiveThreadId,
  onSelectWorkspace,
  onSelectThread,
  onNewThread,
  onLogout,
}: AnythingLLMAppProps) {
  const location = useLocation()
  const navigate = useNavigate()

  const handleLogout = useCallback(() => {
    if (onLogout) {
      onLogout()
    } else {
      try {
        localStorage.removeItem('token')
        localStorage.removeItem('access_token')
      } catch (e) {
        console.error('Logout error:', e)
      }
      navigate('/login')
    }
  }, [onLogout, navigate])

  // User & Auth State
  const [currentUser, setCurrentUser] = useState<{ id: string; email: string; role: 'ADMIN' | 'USER' } | null>(null)
  const [userAccess, setUserAccess] = useState<Record<string, string[]>>({})
  const [authLoading, setAuthLoading] = useState(true)

  // Dynamic Departments & Workspaces
  const [departments, setDepartments] = useState<DepartmentResponse[]>([])
  const [workspaces, setWorkspaces] = useState<WorkspaceItem[]>([])
  const [loadingWorkspaces, setLoadingWorkspaces] = useState(true)
  const [workspaceError, setWorkspaceError] = useState<string | null>(null)

  // Active Navigation & Workspace State
  const [currentNavView, setCurrentNavView] = useState<GlobalNavView>('chat')
  const [activeWorkspace, setActiveWorkspace] = useState<WorkspaceItem | null>(null)
  const [activeWsTab, setActiveWsTab] = useState<WorkspaceTab>('threads')

  // Threads / Conversations State
  const [threads, setThreads] = useState<ConversationItem[]>([])
  const [activeThreadId, setActiveThreadId] = useState<string | null>(propActiveThreadId || null)
  const [loadingThreads, setLoadingThreads] = useState(false)
  const [messages, setMessages] = useState<Array<{ id: string; role: 'user' | 'assistant' | 'system'; content: string; visualization?: VisualizationSpec | null; sources?: Array<{ type?: string; title: string; excerpt?: string; thread_id?: string; thread_title?: string }> }>>([])

  // Documents State
  const [documents, setDocuments] = useState<DocumentListItem[]>([])
  const [loadingDocs, setLoadingDocs] = useState(false)
  const [uploadFile, setUploadFile] = useState<File | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)

  // Modals & Popovers
  const [showCreateDeptModal, setShowCreateDeptModal] = useState(false)
  const [newDeptCode, setNewDeptCode] = useState('')
  const [newDeptName, setNewDeptName] = useState('')
  const [creatingDept, setCreatingDept] = useState(false)
  const [deptModalError, setDeptModalError] = useState<string | null>(null)

  const [showAccessReqModal, setShowAccessReqModal] = useState(false)
  const [accessReqDept, setAccessReqDept] = useState('')
  const [accessReqPerms, setAccessReqPerms] = useState<PermissionCode[]>(['VIEW'])
  const [accessReqReason, setAccessReqReason] = useState('')
  const [submittingAccessReq, setSubmittingAccessReq] = useState(false)
  const [accessRequests, setAccessRequests] = useState<AccessRequestResponse[]>([])

  const [settingsModalWs, setSettingsModalWs] = useState<WorkspaceItem | null>(null)
  const [membersModalWs, setMembersModalWs] = useState<WorkspaceItem | null>(null)

  // Composer & UI State
  const [searchQuery, setSearchQuery] = useState('')
  const [messageInput, setMessageInput] = useState('')
  const [isSending, setIsSending] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [toolsOpen, setToolsOpen] = useState(false)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [toast, setToast] = useState<{ type: 'success' | 'error'; message: string } | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const showToast = useCallback((type: 'success' | 'error', message: string) => {
    setToast({ type, message })
    setTimeout(() => setToast(null), 4000)
  }, [])

  const activeWorkspaceRef = useRef<WorkspaceItem | null>(null)
  activeWorkspaceRef.current = activeWorkspace

  // 1. Fetch User & Access Permissions from Backend
  const loadUserAndAccess = useCallback(async () => {
    if (!token) return null
    try {
      setAuthLoading(true)
      const [me, access] = await Promise.all([
        userApi.getMe(token),
        userApi.getMyAccess(token).catch(() => ({})),
      ])
      setCurrentUser({ id: me.id, email: me.email, role: me.role as 'ADMIN' | 'USER' })
      setUserAccess(access || {})
      return { user: me, access }
    } catch (err: any) {
      console.error('Failed to load user access:', err)
      showToast('error', 'Session expired or invalid token.')
      return null
    } finally {
      setAuthLoading(false)
    }
  }, [token, showToast])

  // 2. Fetch Departments & Workspaces from Backend (NO HARDCODING)
  const loadDepartmentsAndWorkspaces = useCallback(async (role?: 'ADMIN' | 'USER', accessMap?: Record<string, string[]>) => {
    if (!token) return
    setLoadingWorkspaces(true)
    setWorkspaceError(null)
    try {
      const userRole = role || 'USER'
      const userAccessMap = accessMap || {}

      // Backend call for departments
      const depts: DepartmentResponse[] = userRole === 'ADMIN'
        ? await getAllDepartments(token)
        : await getActiveDepartments(token)

      setDepartments(depts)

      // Backend call for workspaces
      const wsList: WorkspaceItem[] = await listWorkspaces(token, undefined, userRole === 'ADMIN')

      // Merge / construct 1:1 Department Workspaces dynamically
      const activeDepts = userRole === 'ADMIN'
        ? depts.filter((d) => d.is_active)
        : depts.filter((d) => d.is_active && (userAccessMap[d.code] || Object.keys(userAccessMap).length === 0))

      const dynamicWorkspaces: WorkspaceItem[] = activeDepts.map((dept) => {
        const existingWs = wsList.find((w) => w.department_id === dept.id || w.department_code === dept.code)
        if (existingWs) {
          return {
            ...existingWs,
            department_name: dept.name,
            department_code: dept.code,
            name: existingWs.name || dept.name,
          }
        }
        return {
          id: dept.id,
          name: dept.name,
          code: dept.code.toLowerCase(),
          department_id: dept.id,
          department_name: dept.name,
          department_code: dept.code,
          is_active: dept.is_active,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        }
      })

      setWorkspaces(dynamicWorkspaces)

      // Restore workspace/thread context from URL or fallback
      const cleanPath = window.location.pathname.replace(/^\/app/, '').replace(/^\/admin/, '')
      const threadM = cleanPath.match(/^\/workspace\/([^\/]+)\/thread\/([^\/]+)/)
      const wsM = cleanPath.match(/^\/workspace\/([^\/]+)/)
      const isChatUrl = cleanPath.startsWith('/chat') || window.location.pathname.includes('/chat')
      const targetWsId = threadM?.[1] || wsM?.[1] || null
      const targetThreadId = threadM?.[2] || null

      let matchedWs: WorkspaceItem | null = null
      if (targetWsId) {
        matchedWs = dynamicWorkspaces.find(
          (w) =>
            w.id === targetWsId ||
            w.department_id === targetWsId ||
            w.code.toLowerCase() === targetWsId.toLowerCase() ||
            w.department_code?.toLowerCase() === targetWsId.toLowerCase()
        ) || null
      }

      if (matchedWs) {
        setActiveWorkspace(matchedWs)
        setCurrentNavView('chat')
        setActiveWsTab('threads')
        if (targetThreadId) {
          setActiveThreadId(targetThreadId)
        }
      } else if (isChatUrl) {
        setActiveWorkspace(null)
        setCurrentNavView('chat')
      } else if (dynamicWorkspaces.length > 0) {
        if (!activeWorkspaceRef.current || !dynamicWorkspaces.some((w) => w.id === activeWorkspaceRef.current?.id)) {
          setActiveWorkspace(dynamicWorkspaces[0])
        }
      } else {
        setActiveWorkspace(null)
      }
    } catch (err: any) {
      console.error('Failed to load workspaces:', err)
      setWorkspaceError(err.message || 'Unable to load workspaces from backend.')
    } finally {
      setLoadingWorkspaces(false)
    }
  }, [token])

  // Initial load
  useEffect(() => {
    loadUserAndAccess().then((res) => {
      if (res) {
        loadDepartmentsAndWorkspaces(res.user.role as 'ADMIN' | 'USER', res.access)
      }
    })
  }, [token])

  const activeThreadIdRef = useRef<string | null>(activeThreadId)
  useEffect(() => {
    activeThreadIdRef.current = activeThreadId
  }, [activeThreadId])

  // Helper: Deduplicate messages by stable message ID
  const deduplicateMessages = useCallback(<T extends { id: string | number }>(msgs: T[]): T[] => {
    const seen = new Set<string | number>()
    const result: T[] = []
    for (const m of msgs) {
      if (m.id && !seen.has(m.id)) {
        seen.add(m.id)
        result.push(m)
      }
    }
    return result
  }, [])

  // 3. Load Scoped Threads for Active Workspace or Global Chat
  const loadThreadsForWorkspace = useCallback(async (wsId?: string | null) => {
    if (!token) return
    setLoadingThreads(true)
    try {
      const convs = await listConversations(token, wsId || undefined)
      setThreads(convs)
      const currentId = activeThreadIdRef.current
      if (convs.length > 0) {
        if (!currentId || !convs.some((c) => c.id === currentId)) {
          setActiveThreadId(convs[0].id)
        }
      } else {
        setActiveThreadId(null)
        setMessages([])
      }
    } catch (err: any) {
      console.error('Failed to load threads:', err)
    } finally {
      setLoadingThreads(false)
    }
  }, [token])

  useEffect(() => {
    if (activeWorkspace) {
      loadThreadsForWorkspace(activeWorkspace.id)
    } else if (currentNavView === 'chat') {
      loadThreadsForWorkspace(null)
    }
  }, [activeWorkspace, currentNavView, loadThreadsForWorkspace])

  // 4. Load Conversation Messages when Thread Changes (with stable deduplication)
  useEffect(() => {
    if (!token || !activeThreadId) {
      setMessages([])
      return
    }
    let isCancelled = false
    getConversation(activeThreadId, token)
      .then((detail) => {
        if (isCancelled) return
        if (detail && detail.messages) {
          const loaded = detail.messages.map((m: any) => ({
            id: String(m.id),
            role: m.role,
            content: m.content,
            visualization: m.visualization,
            sources: m.sources,
          }))
          setMessages(deduplicateMessages(loaded))
        }
      })
      .catch((err) => {
        if (!isCancelled) console.error('Failed to load conversation messages:', err)
      })
    return () => {
      isCancelled = true
    }
  }, [token, activeThreadId, deduplicateMessages])

  // 5. Load Documents for Active View / Tab
  const loadDocuments = useCallback(async () => {
    if (!token) return
    setLoadingDocs(true)
    try {
      const data = await documentApi.listDocuments(token)
      setDocuments(data)
    } catch (err: any) {
      console.error('Failed to load documents:', err)
    } finally {
      setLoadingDocs(false)
    }
  }, [token])

  useEffect(() => {
    if (currentNavView === 'documents' || activeWsTab === 'documents') {
      loadDocuments()
    }
  }, [currentNavView, activeWsTab, loadDocuments])

  // 6. Load Access Requests
  const loadAccessRequests = useCallback(async () => {
    if (!token) return
    try {
      const reqs = await accessRequestApi.getMyAccessRequests(token)
      setAccessRequests(reqs)
    } catch (err) {
      console.error('Failed to load access requests:', err)
    }
  }, [token])

  useEffect(() => {
    if (currentNavView === 'requests') {
      loadAccessRequests()
    }
  }, [currentNavView, loadAccessRequests])

  // Scroll to bottom of chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isSending])

  // Permissions helper for active workspace
  const getDeptPermissions = (deptCode?: string | null): string[] => {
    if (currentUser?.role === 'ADMIN') return ['VIEW', 'UPLOAD', 'EDIT', 'DELETE']
    if (!deptCode) return []
    return userAccess[deptCode.toUpperCase()] || []
  }

  const activeDeptCode = activeWorkspace?.department_code || activeWorkspace?.code
  const currentPermissions = getDeptPermissions(activeDeptCode)
  const canUpload = currentPermissions.includes('UPLOAD') || currentPermissions.includes('ALL') || currentUser?.role === 'ADMIN'
  const canDelete = currentPermissions.includes('DELETE') || currentPermissions.includes('ALL') || currentUser?.role === 'ADMIN'

  // Chat message sending with department & workspace scoping
  const handleSendMessage = async () => {
    if (!messageInput.trim() || isSending) return
    const queryText = messageInput.trim()
    const tempUserMsgId = `user-${Date.now()}`
    const userMsg = { id: tempUserMsgId, role: 'user' as const, content: queryText }
    setMessages((prev) => deduplicateMessages([...prev, userMsg]))
    setMessageInput('')
    setIsSending(true)

    try {
      const currentThreadId = activeThreadIdRef.current
      const isTypingVisualize = queryText.toLowerCase() === 'visualize this' || queryText.toLowerCase() === 'visulaize this'
      const lastAsst = [...messages].reverse().find((m) => m.role === 'assistant')
      const targetMsgId = isTypingVisualize && lastAsst ? lastAsst.id : undefined
      const actionVal = isTypingVisualize ? 'visualize' : undefined

      const response = await askQuestion(
        {
          query: isTypingVisualize ? 'Visualize this' : queryText,
          conversation_id: currentThreadId || undefined,
          workspace_id: activeWorkspace?.id || undefined,
          action: actionVal,
          target_message_id: targetMsgId,
        },
        token
      )

      const rawSources = (response as any).sources || (response.retrieved_chunks ? response.retrieved_chunks.map((c: any) => ({
        type: 'document',
        title: `${c.file_name}${c.page_number ? ` (Page ${c.page_number})` : ''}`,
        excerpt: c.text
      })) : [])

      const assistantMsgId = response.message_id ? String(response.message_id) : `asst-${Date.now()}`
      const assistantMsg = {
        id: assistantMsgId,
        role: 'assistant' as const,
        content: response.answer || 'Response generated from workspace documents.',
        visualization: response.visualization,
        sources: rawSources,
      }
      setMessages((prev) => deduplicateMessages([...prev, assistantMsg]))

      // Requirement 3: Automatic title generated on backend -> immediately update sidebar
      if (response.conversation_title) {
        const genTitle = response.conversation_title
        setThreads((prev) =>
          prev.map((t) =>
            t.id === (response.conversation_id || currentThreadId)
              ? { ...t, title: genTitle }
              : t
          )
        )
      }

      // If this was a new thread, activate it and refresh thread list to display generated title
      if (!currentThreadId && response.conversation_id) {
        setActiveThreadId(response.conversation_id)
        if (activeWorkspace) {
          navigate(`/app/workspace/${activeWorkspace.id}/thread/${response.conversation_id}`, { replace: true })
          await loadThreadsForWorkspace(activeWorkspace.id)
        } else {
          navigate(`/app/chat/thread/${response.conversation_id}`, { replace: true })
          await loadThreadsForWorkspace(null)
        }
      }
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        {
          id: `err-${Date.now()}`,
          role: 'assistant' as const,
          content: `Error: ${err.message || 'Failed to retrieve answer from workspace.'}`,
        },
      ])
    } finally {
      setIsSending(false)
    }
  }

  const handleVisualizeMessage = async (targetMessageId: string) => {
    if (isSending) return
    setIsSending(true)
    try {
      const currentThreadId = activeThreadIdRef.current
      const response = await askQuestion(
        {
          query: 'Visualize this',
          action: 'visualize',
          target_message_id: targetMessageId,
          conversation_id: currentThreadId || undefined,
          workspace_id: activeWorkspace?.id || undefined,
        },
        token
      )

      const rawSources = (response as any).sources || (response.retrieved_chunks ? response.retrieved_chunks.map((c: any) => ({
        type: 'document',
        title: `${c.file_name}${c.page_number ? ` (Page ${c.page_number})` : ''}`,
        excerpt: c.text
      })) : [])

      const assistantMsgId = response.message_id ? String(response.message_id) : `asst-${Date.now()}`
      const assistantMsg = {
        id: assistantMsgId,
        role: 'assistant' as const,
        content: response.answer || 'Response generated from workspace documents.',
        visualization: response.visualization,
        sources: rawSources,
      }
      setMessages((prev) => deduplicateMessages([...prev, assistantMsg]))
    } catch (err: any) {
      console.error('Visualization error:', err)
    } finally {
      setIsSending(false)
    }
  }

  const handleCreateNewThreadForWs = async (targetWs: WorkspaceItem) => {
    setActiveWorkspace(targetWs)
    try {
      const newConv = await createConversation('New Thread', token, targetWs.id)
      const realWsId = newConv.workspace_id || targetWs.id
      const updatedWs = { ...targetWs, id: realWsId }
      setActiveWorkspace(updatedWs)
      setWorkspaces((prev) =>
        prev.map((w) =>
          w.id === targetWs.id || w.department_id === targetWs.department_id
            ? { ...w, id: realWsId }
            : w
        )
      )
      setThreads((prev) => [newConv, ...prev.filter((t) => t.id !== newConv.id)])
      setActiveThreadId(newConv.id)
      setMessages([])
      setActiveWsTab('threads')
      setCurrentNavView('chat')
      navigate(`/app/workspace/${realWsId}/thread/${newConv.id}`)
      onSelectWorkspace?.(realWsId)
      onSelectThread?.(newConv.id, realWsId)
      onNewThread?.(realWsId)
      showToast('success', `Created new thread in ${targetWs.name}`)
    } catch (err: any) {
      console.error('Failed to create new thread:', err)
      showToast('error', err.message || 'Failed to create new thread.')
      setActiveThreadId(null)
      setMessages([])
      setActiveWsTab('threads')
      setCurrentNavView('chat')
    }
  }

  const handleCreateNewThread = async () => {
    if (activeWorkspace) {
      await handleCreateNewThreadForWs(activeWorkspace)
    } else {
      try {
        const newThread = await createConversation('New Thread', token, null)
        setThreads((prev) => [newThread, ...prev])
        setActiveThreadId(newThread.id)
        setMessages([])
        setActiveWsTab('threads')
        setCurrentNavView('chat')
        showToast('success', 'Created new Global Chat thread.')
      } catch (err: any) {
        showToast('error', err.message || 'Failed to create new thread.')
      }
    }
  }

  const handleDeleteThread = async (e: React.MouseEvent, threadId: string) => {
    e.stopPropagation()
    try {
      await deleteConversation(threadId, token)
      setThreads((prev) => prev.filter((t) => t.id !== threadId))
      if (activeThreadId === threadId) {
        setActiveThreadId(null)
        setMessages([])
      }
      showToast('success', 'Thread deleted.')
    } catch (err: any) {
      showToast('error', err.message || 'Failed to delete thread.')
    }
  }

  // Document Upload
  const handleUploadDocument = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!uploadFile || !activeDeptCode) {
      setUploadError('Please select a file to upload.')
      return
    }
    setIsUploading(true)
    setUploadError(null)
    setUploadSuccess(null)
    try {
      await uploadDocument(uploadFile, activeDeptCode, token, activeWorkspace?.id)
      setUploadSuccess(`Uploaded ${uploadFile.name} successfully.`)
      setUploadFile(null)
      loadDocuments()
      showToast('success', `Uploaded ${uploadFile.name} to ${activeWorkspace?.name}`)
    } catch (err: any) {
      setUploadError(err.message || 'Upload failed. Check your department permissions.')
      showToast('error', err.message || 'Upload failed.')
    } finally {
      setIsUploading(false)
    }
  }

  // Create Department (Admin Only)
  const handleCreateDepartment = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!newDeptCode.trim() || !newDeptName.trim()) {
      setDeptModalError('Department code and name are required.')
      return
    }
    setCreatingDept(true)
    setDeptModalError(null)
    try {
      await createDepartment(newDeptCode.trim().toUpperCase(), newDeptName.trim(), token)
      showToast('success', `Department ${newDeptName} created successfully!`)
      setShowCreateDeptModal(false)
      setNewDeptCode('')
      setNewDeptName('')
      // Reload departments to immediately reflect in WORKSPACES
      await loadDepartmentsAndWorkspaces()
    } catch (err: any) {
      setDeptModalError(err.message || 'Failed to create department.')
    } finally {
      setCreatingDept(false)
    }
  }

  // Submit Access Request
  const handleSubmitAccessRequest = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!accessReqDept) {
      showToast('error', 'Please select a department.')
      return
    }
    setSubmittingAccessReq(true)
    try {
      await accessRequestApi.createAccessRequest(
        {
          department_code: accessReqDept,
          requested_permissions: accessReqPerms,
          reason: accessReqReason,
        },
        token
      )
      showToast('success', `Access request for ${accessReqDept} submitted!`)
      setShowAccessReqModal(false)
      setAccessReqReason('')
      loadAccessRequests()
    } catch (err: any) {
      showToast('error', err.message || 'Failed to submit access request.')
    } finally {
      setSubmittingAccessReq(false)
    }
  }

  // Filtered workspaces for search
  const filteredWorkspaces = workspaces.filter((ws) =>
    ws.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    (ws.department_code && ws.department_code.toLowerCase().includes(searchQuery.toLowerCase()))
  )

  // Scoped documents for active department
  const workspaceDocs = documents.filter((d) =>
    d.department_code?.toUpperCase() === activeDeptCode?.toUpperCase() ||
    d.department_id === activeWorkspace?.department_id ||
    d.workspace_id === activeWorkspace?.id
  )

  return (
    <div
      style={{
        display: 'flex',
        width: '100vw',
        height: '100vh',
        overflow: 'hidden',
        background: '#FFFFFF',
        fontFamily: 'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
        color: '#252525',
      }}
    >
      {/* Toast Notification */}
      {toast && (
        <div
          style={{
            position: 'fixed',
            top: '16px',
            right: '20px',
            zIndex: 9999,
            padding: '10px 16px',
            borderRadius: '8px',
            fontSize: '12px',
            fontWeight: 500,
            boxShadow: '0 4px 16px rgba(0,0,0,0.12)',
            background: toast.type === 'error' ? '#FEF2F2' : '#F0FDF4',
            border: `1px solid ${toast.type === 'error' ? '#FECACA' : '#BBF7D0'}`,
            color: toast.type === 'error' ? '#991B1B' : '#166534',
          }}
        >
          {toast.message}
        </div>
      )}

      {/* ========================================================================= */}
      {/* LEFT SIDEBAR (~148px)                                                     */}
      {/* ========================================================================= */}
      {!sidebarCollapsed && (
        <aside
          style={{
            width: '148px',
            minWidth: '148px',
            maxWidth: '148px',
            height: '100vh',
            background: '#F7F7F7',
            borderRight: '1px solid #E5E5E5',
            display: 'flex',
            flexDirection: 'column',
            padding: '10px 8px 8px 8px',
            boxSizing: 'border-box',
            userSelect: 'none',
            flexShrink: 0,
          }}
        >
          {/* 1. Top Brand Header */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              marginBottom: '10px',
              padding: '0 2px',
            }}
          >
            <div
              style={{ display: 'flex', alignItems: 'center', gap: '5px', cursor: 'pointer' }}
              onClick={() => setCurrentNavView('chat')}
            >
              {/* Loop Logo */}
              <svg
                width="15"
                height="15"
                viewBox="0 0 24 24"
                fill="none"
                stroke="#1F2937"
                strokeWidth="2.2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M18.178 8c5.096 0 5.096 8 0 8-5.095 0-7.133-8-12.356-8C.726 8 .726 16 5.822 16c5.095 0 7.133-8 12.356-8z" />
              </svg>
              <span
                style={{
                  fontSize: '12.5px',
                  fontWeight: 600,
                  color: '#1F2937',
                  letterSpacing: '-0.2px',
                  whiteSpace: 'nowrap',
                }}
              >
                RAG Workspace
              </span>
            </div>

            <button
              onClick={() => setSidebarCollapsed(true)}
              title="Collapse sidebar"
              style={{
                background: 'transparent',
                border: 'none',
                cursor: 'pointer',
                padding: '2px',
                color: '#9CA3AF',
                display: 'flex',
                alignItems: 'center',
              }}
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                <line x1="9" y1="3" x2="9" y2="21" />
              </svg>
            </button>
          </div>

          {/* 2. Search Bar */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              background: '#FFFFFF',
              border: '1px solid #E5E7EB',
              borderRadius: '6px',
              padding: '4px 6px',
              height: '28px',
              marginBottom: '8px',
              gap: '4px',
              boxSizing: 'border-box',
            }}
          >
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#9CA3AF" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="11" cy="11" r="8" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
            </svg>
            <input
              type="text"
              placeholder="Search"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={{
                border: 'none',
                outline: 'none',
                fontSize: '11px',
                width: '100%',
                color: '#374151',
                background: 'transparent',
                padding: 0,
              }}
            />
            {currentUser?.role === 'ADMIN' && (
              <button
                onClick={() => setShowCreateDeptModal(true)}
                title="Create Department"
                style={{
                  background: 'transparent',
                  border: 'none',
                  padding: 0,
                  cursor: 'pointer',
                  color: '#9CA3AF',
                  display: 'flex',
                  alignItems: 'center',
                }}
              >
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="12" y1="5" x2="12" y2="19" />
                  <line x1="5" y1="12" x2="19" y2="12" />
                </svg>
              </button>
            )}
          </div>

          {/* 3. Global Navigation Section */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '2px', marginBottom: '8px' }}>
            {[
              {
                id: 'chat',
                label: 'Global Chat',
                icon: (
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                  </svg>
                ),
              },
              {
                id: 'documents',
                label: 'Documents',
                icon: (
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                    <polyline points="14 2 14 8 20 8" />
                    <line x1="16" y1="13" x2="8" y2="13" />
                    <line x1="16" y1="17" x2="8" y2="17" />
                  </svg>
                ),
              },
              {
                id: 'knowledge',
                label: 'Knowledge',
                icon: (
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="12" cy="12" r="3" />
                    <circle cx="6" cy="6" r="2" />
                    <circle cx="18" cy="6" r="2" />
                    <circle cx="6" cy="18" r="2" />
                    <circle cx="18" cy="18" r="2" />
                    <line x1="7.5" y1="7.5" x2="10" y2="10" />
                    <line x1="16.5" y1="7.5" x2="14" y2="10" />
                    <line x1="7.5" y1="16.5" x2="10" y2="14" />
                    <line x1="16.5" y1="16.5" x2="14" y2="14" />
                  </svg>
                ),
              },
              ...(currentUser?.role === 'ADMIN'
                ? [
                    {
                      id: 'departments',
                      label: 'Departments',
                      icon: (
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <rect x="4" y="2" width="16" height="20" rx="2" ry="2" />
                          <line x1="9" y1="22" x2="9" y2="22.01" />
                          <line x1="15" y1="22" x2="15" y2="22.01" />
                          <line x1="8" y1="6" x2="8" y2="6.01" />
                          <line x1="16" y1="6" x2="16" y2="6.01" />
                          <line x1="8" y1="10" x2="8" y2="10.01" />
                          <line x1="16" y1="10" x2="16" y2="10.01" />
                        </svg>
                      ),
                    },
                  ]
                : []),
              {
                id: 'requests',
                label: 'Requests',
                icon: (
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z" />
                    <polyline points="22,6 12,13 2,6" />
                  </svg>
                ),
              },
            ].map((nav) => {
              const isSelected = nav.id === 'chat'
                ? (currentNavView === 'chat' && activeWorkspace === null)
                : (currentNavView === nav.id)
              return (
                <div
                  key={nav.id}
                  onClick={() => {
                    if (nav.id === 'chat') {
                      setCurrentNavView('chat')
                      setActiveWorkspace(null)
                      setActiveThreadId(null)
                      setMessages([])
                      navigate('/app/chat')
                    } else {
                      setCurrentNavView(nav.id as GlobalNavView)
                    }
                  }}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    padding: '4px 6px',
                    borderRadius: '5px',
                    fontSize: '11px',
                    fontWeight: isSelected ? 600 : 400,
                    color: isSelected ? '#1E3A8A' : '#4B5563',
                    background: isSelected ? '#E5EFFF' : 'transparent',
                    cursor: 'pointer',
                    transition: 'background 0.12s ease',
                  }}
                  onMouseEnter={(e) => {
                    if (!isSelected) e.currentTarget.style.background = '#EEEEEE'
                  }}
                  onMouseLeave={(e) => {
                    if (!isSelected) e.currentTarget.style.background = 'transparent'
                  }}
                >
                  <span style={{ display: 'flex', alignItems: 'center', color: isSelected ? '#1E3A8A' : '#6B7280' }}>
                    {nav.icon}
                  </span>
                  <span style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{nav.label}</span>
                </div>
              )
            })}
          </div>

          {/* 4. WORKSPACES Header */}
          <div
            style={{
              fontSize: '9.5px',
              fontWeight: 700,
              color: '#9CA3AF',
              letterSpacing: '0.06em',
              textTransform: 'uppercase',
              padding: '4px 4px 2px 4px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}
          >
            <span>Workspaces</span>
            {currentUser?.role === 'ADMIN' && (
              <span
                onClick={() => setShowCreateDeptModal(true)}
                title="Add department workspace"
                style={{ cursor: 'pointer', fontSize: '11px', color: '#6B7280' }}
              >
                +
              </span>
            )}
          </div>

          {/* 5. Dynamically Generated Workspaces (Departments) */}
          <div
            style={{
              flex: 1,
              overflowY: 'auto',
              overflowX: 'hidden',
              display: 'flex',
              flexDirection: 'column',
              gap: '5px',
              paddingTop: '2px',
            }}
          >
            {loadingWorkspaces ? (
              <div style={{ fontSize: '10.5px', color: '#9CA3AF', padding: '6px 4px' }}>Loading workspaces…</div>
            ) : workspaceError ? (
              <div style={{ padding: '6px 4px' }}>
                <div style={{ fontSize: '10.5px', color: '#DC2626' }}>{workspaceError}</div>
                <button
                  onClick={() => loadDepartmentsAndWorkspaces()}
                  style={{ fontSize: '10px', marginTop: '4px', background: 'transparent', border: '1px solid #E5E7EB', borderRadius: '4px', cursor: 'pointer', padding: '2px 6px' }}
                >
                  Retry
                </button>
              </div>
            ) : filteredWorkspaces.length === 0 ? (
              <div style={{ fontSize: '10.5px', color: '#9CA3AF', padding: '8px 4px', lineHeight: 1.4 }}>
                {currentUser?.role === 'ADMIN' ? (
                  <>
                    No departments yet.
                    <button
                      onClick={() => setShowCreateDeptModal(true)}
                      style={{ display: 'block', marginTop: '4px', fontSize: '10px', padding: '2px 6px', background: '#E5EFFF', border: '1px solid #BFDBFE', borderRadius: '4px', cursor: 'pointer', color: '#1E3A8A' }}
                    >
                      + Create
                    </button>
                  </>
                ) : (
                  'No department access.'
                )}
              </div>
            ) : (
              filteredWorkspaces.map((ws) => {
                const isActiveWs = activeWorkspace?.id === ws.id
                return (
                  <div key={ws.id} style={{ display: 'flex', flexDirection: 'column' }}>
                    {/* Workspace Card Header */}
                    <div
                      onClick={() => {
                        if (activeWorkspace?.id !== ws.id) {
                          setActiveWorkspace(ws)
                          setActiveThreadId(null)
                          setMessages([])
                        }
                        setCurrentNavView('chat')
                        setActiveWsTab('threads')
                        navigate(`/app/workspace/${ws.id}`)
                        onSelectWorkspace?.(ws.id)
                      }}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        padding: '4px 6px',
                        borderRadius: '5px',
                        background: isActiveWs ? '#DCEAFE' : '#FFFFFF',
                        border: isActiveWs ? '1px solid #BFDBFE' : '1px solid #E5E7EB',
                        cursor: 'pointer',
                        boxShadow: isActiveWs ? 'none' : '0 1px 2px rgba(0, 0, 0, 0.02)',
                        transition: 'all 0.15s ease',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '4px', minWidth: 0, flex: 1 }}>
                        <svg width="7" height="9" viewBox="0 0 10 16" fill="#9CA3AF" style={{ flexShrink: 0 }}>
                          <circle cx="3" cy="3" r="1.5" />
                          <circle cx="7" cy="3" r="1.5" />
                          <circle cx="3" cy="8" r="1.5" />
                          <circle cx="7" cy="8" r="1.5" />
                          <circle cx="3" cy="13" r="1.5" />
                          <circle cx="7" cy="13" r="1.5" />
                        </svg>
                        <span
                          style={{
                            fontSize: '11px',
                            fontWeight: isActiveWs ? 600 : 500,
                            color: isActiveWs ? '#1E3A8A' : '#374151',
                            whiteSpace: 'nowrap',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                          }}
                        >
                          {ws.name}
                        </span>
                      </div>

                      {/* Icons for active workspace */}
                      {isActiveWs && (
                        <div style={{ display: 'flex', alignItems: 'center', gap: '2px', flexShrink: 0 }}>
                          {canUpload && (
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                setActiveWsTab('documents')
                              }}
                              title="Upload to department"
                              style={{ background: 'transparent', border: 'none', cursor: 'pointer', padding: 0, color: '#6B7280', display: 'flex' }}
                            >
                              <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
                                <polyline points="16 9 12 5 8 9" />
                                <line x1="12" y1="5" x2="12" y2="15" />
                              </svg>
                            </button>
                          )}
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              setSettingsModalWs(ws)
                            }}
                            title="Workspace settings"
                            style={{ background: 'transparent', border: 'none', cursor: 'pointer', padding: 0, color: '#6B7280', display: 'flex' }}
                          >
                            <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                              <circle cx="12" cy="12" r="3" />
                              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
                            </svg>
                          </button>
                        </div>
                      )}
                    </div>

                    {/* Department Threads Tree */}
                    {isActiveWs && (
                      <div
                        style={{
                          marginLeft: '8px',
                          paddingLeft: '7px',
                          borderLeft: '1.5px solid #CBD5E1',
                          display: 'flex',
                          flexDirection: 'column',
                          gap: '2px',
                          marginTop: '3px',
                          marginBottom: '3px',
                        }}
                      >
                        {loadingThreads ? (
                          <div style={{ fontSize: '10px', color: '#9CA3AF', padding: '2px 4px' }}>Loading…</div>
                        ) : threads.length === 0 ? (
                          <div style={{ fontSize: '10px', color: '#9CA3AF', padding: '2px 4px' }}>No threads</div>
                        ) : (
                          threads.map((thread) => {
                            const isSelectedThread = thread.id === activeThreadId
                            return (
                              <div
                                key={thread.id}
                                onClick={() => {
                                  setActiveThreadId(thread.id)
                                  setActiveWsTab('threads')
                                  setCurrentNavView('chat')
                                  navigate(`/app/workspace/${ws.id}/thread/${thread.id}`)
                                  onSelectThread?.(thread.id, ws.id)
                                }}
                                style={{
                                  display: 'flex',
                                  alignItems: 'center',
                                  justifyContent: 'space-between',
                                  padding: '2px 5px',
                                  borderRadius: '4px',
                                  fontSize: '10.5px',
                                  fontWeight: isSelectedThread ? 500 : 400,
                                  color: isSelectedThread ? '#1E3A8A' : '#4B5563',
                                  background: isSelectedThread ? '#CCE0FC' : 'transparent',
                                  cursor: 'pointer',
                                  whiteSpace: 'nowrap',
                                  overflow: 'hidden',
                                  textOverflow: 'ellipsis',
                                }}
                              >
                                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{thread.title}</span>
                                {isSelectedThread && (
                                  <span
                                    onClick={(e) => handleDeleteThread(e, thread.id)}
                                    title="Delete thread"
                                    style={{ fontSize: '10px', color: '#9CA3AF', marginLeft: '4px' }}
                                  >
                                    ×
                                  </span>
                                )}
                              </div>
                            )
                          })
                        )}

                        {/* + New Thread Button */}
                        <div
                          onClick={(e) => {
                            e.stopPropagation()
                            handleCreateNewThreadForWs(ws)
                          }}
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '3px',
                            padding: '2px 4px',
                            marginTop: '2px',
                            fontSize: '10px',
                            fontWeight: 500,
                            color: '#4B5563',
                            cursor: 'pointer',
                            borderRadius: '3px',
                          }}
                        >
                          <span style={{ fontSize: '11px', fontWeight: 700, color: '#6B7280' }}>+</span>
                          <span>New Thread</span>
                        </div>
                      </div>
                    )}
                  </div>
                )
              })
            )}
          </div>

          {/* 6. Sidebar Bottom Toolbar */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '8px 4px 2px 4px',
              borderTop: '1px solid #EBEBEB',
              marginTop: 'auto',
            }}
          >
            <button
              onClick={() => {
                if (activeWorkspace) setSettingsModalWs(activeWorkspace)
              }}
              title="Settings"
              style={{ background: 'transparent', border: 'none', cursor: 'pointer', padding: '2px', color: '#6B7280' }}
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="3" />
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
              </svg>
            </button>
            <button
              onClick={() => setShowAccessReqModal(true)}
              title="Request Access"
              style={{ background: 'transparent', border: 'none', cursor: 'pointer', padding: '2px', color: '#6B7280' }}
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
                <circle cx="8.5" cy="7" r="4" />
                <line x1="20" y1="8" x2="20" y2="14" />
                <line x1="23" y1="11" x2="17" y2="11" />
              </svg>
            </button>
            <button
              onClick={() => {
                if (activeWorkspace) setMembersModalWs(activeWorkspace)
              }}
              title="Members & Roles"
              style={{ background: 'transparent', border: 'none', cursor: 'pointer', padding: '2px', color: '#6B7280' }}
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
                <circle cx="9" cy="7" r="4" />
                <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
                <path d="M16 3.13a4 4 0 0 1 0 7.75" />
              </svg>
            </button>
            <button
              onClick={handleLogout}
              title="Log out"
              style={{ background: 'transparent', border: 'none', cursor: 'pointer', padding: '2px', color: '#6B7280' }}
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
                <polyline points="16 17 21 12 16 7" />
                <line x1="21" y1="12" x2="9" y2="12" />
              </svg>
            </button>
          </div>
        </aside>
      )}

      {/* ========================================================================= */}
      {/* MAIN CONTENT AREA                                                         */}
      {/* ========================================================================= */}
      <main
        style={{
          flex: 1,
          height: '100vh',
          background: '#FFFFFF',
          display: 'flex',
          flexDirection: 'column',
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        {/* Top Header Bar */}
        <header
          style={{
            height: '38px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '0 16px',
            borderBottom: '1px solid #F0F0F0',
            boxSizing: 'border-box',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            {sidebarCollapsed && (
              <button
                onClick={() => setSidebarCollapsed(false)}
                title="Expand sidebar"
                style={{ background: 'transparent', border: 'none', cursor: 'pointer', padding: '2px', color: '#6B7280', display: 'flex' }}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                  <line x1="9" y1="3" x2="9" y2="21" />
                </svg>
              </button>
            )}

            {/* Department Workspace Title or Global Chat */}
            {activeWorkspace ? (
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ fontSize: '13px', fontWeight: 600, color: '#111827' }}>
                  {activeWorkspace.name}
                </span>
                <span
                  style={{
                    fontSize: '9.5px',
                    padding: '1px 6px',
                    borderRadius: '10px',
                    background: '#F3F4F6',
                    color: '#6B7280',
                    fontFamily: 'monospace',
                  }}
                >
                  {activeWorkspace.department_code || 'DEPT'}
                </span>
              </div>
            ) : (
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ fontSize: '13px', fontWeight: 600, color: '#1E3A8A' }}>
                  Global Chat
                </span>
                <span
                  style={{
                    fontSize: '9.5px',
                    padding: '1px 6px',
                    borderRadius: '10px',
                    background: '#E5EFFF',
                    color: '#1E3A8A',
                    fontFamily: 'monospace',
                    fontWeight: 600,
                  }}
                >
                  {currentUser?.role === 'ADMIN' ? 'GLOBAL · ADMIN' : 'GLOBAL'}
                </span>
              </div>
            )}
          </div>

          {/* Workspace Tabs (Threads | Documents | Knowledge | Members | Provider) */}
          {activeWorkspace && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              {(['threads', 'documents', 'knowledge', 'members', 'provider'] as WorkspaceTab[]).map((tab) => {
                const isTabActive = activeWsTab === tab
                return (
                  <button
                    key={tab}
                    onClick={() => setActiveWsTab(tab)}
                    style={{
                      background: isTabActive ? '#F1F5F9' : 'transparent',
                      border: 'none',
                      borderRadius: '4px',
                      padding: '3px 8px',
                      fontSize: '11px',
                      fontWeight: isTabActive ? 600 : 400,
                      color: isTabActive ? '#1E293B' : '#64748B',
                      cursor: 'pointer',
                      textTransform: 'capitalize',
                      transition: 'all 0.1s ease',
                    }}
                  >
                    {tab === 'threads' ? 'Chat / Threads' : tab === 'provider' ? 'Provider Config' : tab}
                  </button>
                )
              })}
            </div>
          )}

          {/* Top-Right Control (Settings Button) */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ fontSize: '10.5px', color: '#9CA3AF', fontFamily: 'monospace' }}>
              qwen/qwen3.5-9b
            </span>
            <button
              onClick={() => {
                if (activeWorkspace) setSettingsModalWs(activeWorkspace)
              }}
              title="Workspace configuration"
              style={{ background: 'transparent', border: 'none', cursor: 'pointer', padding: '4px', color: '#6B7280', display: 'flex' }}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="4" y1="21" x2="4" y2="14" />
                <line x1="4" y1="10" x2="4" y2="3" />
                <line x1="12" y1="21" x2="12" y2="12" />
                <line x1="12" y1="8" x2="12" y2="3" />
                <line x1="20" y1="21" x2="20" y2="16" />
                <line x1="20" y1="12" x2="20" y2="3" />
                <line x1="1" y1="14" x2="7" y2="14" />
                <line x1="9" y1="8" x2="15" y2="8" />
                <line x1="17" y1="16" x2="23" y2="16" />
              </svg>
            </button>
          </div>
        </header>

        {/* Scrollable View Content */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '16px 28px 95px 28px', boxSizing: 'border-box' }}>
          {/* TAB 1: THREADS / CHAT VIEW */}
          {activeWsTab === 'threads' && currentNavView === 'chat' && (
            <div style={{ maxWidth: '820px', margin: '0 auto', fontSize: '11.5px', lineHeight: 1.6, color: '#252525' }}>
              {/* If no conversation messages yet, render the workspace reference overview */}
              {messages.length === 0 ? (
                <div
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    minHeight: '340px',
                    textAlign: 'center',
                    padding: '40px 16px',
                  }}
                >
                  <div
                    style={{
                      width: '40px',
                      height: '40px',
                      borderRadius: '10px',
                      background: '#F1F5F9',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      marginBottom: '12px',
                    }}
                  >
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#64748B" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                    </svg>
                  </div>
                  <h2
                    style={{
                      fontSize: '18px',
                      fontWeight: 600,
                      color: '#111827',
                      margin: '0 0 6px 0',
                    }}
                  >
                    {activeWorkspace ? activeWorkspace.name : 'Admin Global Chat'}
                  </h2>
                  <p
                    style={{
                      fontSize: '12px',
                      color: '#64748B',
                      maxWidth: '460px',
                      margin: '0 0 16px 0',
                      lineHeight: 1.5,
                    }}
                  >
                    {activeWorkspace
                      ? `Send a message to ask questions or search across documents in this workspace. All answers are grounded in ${activeWorkspace.name} knowledge.`
                      : 'Ask questions across all departmental documents and knowledge (Sales, HR, Finance, etc.). Answers are consolidated using cross-department RAG retrieval.'}
                  </p>
                  <div
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '6px',
                      padding: '4px 10px',
                      borderRadius: '16px',
                      background: '#F8FAFC',
                      border: '1px solid #E2E8F0',
                      fontSize: '11px',
                      color: '#475569',
                    }}
                  >
                    {activeWorkspace ? (
                      <>
                        <span>Department: <strong>{activeDeptCode || 'GLOBAL'}</strong></span>
                        <span>·</span>
                        <span>Permissions: <strong>{currentPermissions.join(', ') || 'VIEW'}</strong></span>
                      </>
                    ) : (
                      <>
                        <span>Scope: <strong>All Active Departments</strong></span>
                        <span>·</span>
                        <span>Role: <strong>Admin Access</strong></span>
                      </>
                    )}
                  </div>
                </div>
              ) : (
                /* Live Conversation Messages */
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {messages.map((m) => (
                    <div
                      key={m.id}
                      style={{
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '4px',
                        padding: '8px 12px',
                        borderRadius: '8px',
                        background: m.role === 'user' ? '#F3F4F6' : '#FFFFFF',
                        border: m.role === 'user' ? '1px solid #E5E7EB' : '1px solid #E2E8F0',
                        alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
                        maxWidth: '85%',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '4px' }}>
                        <span style={{ fontSize: '10px', fontWeight: 600, color: m.role === 'user' ? '#4B5563' : '#1E3A8A', textTransform: 'uppercase' }}>
                          {m.role === 'user' ? 'You' : `${activeWorkspace?.name || 'Workspace'} Assistant`}
                        </span>
                        {m.role === 'assistant' && (
                          <button
                            onClick={() => handleVisualizeMessage(m.id)}
                            title="Visualize this response"
                            style={{
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: '4px',
                              padding: '2px 7px',
                              fontSize: '10.5px',
                              fontWeight: 600,
                              color: '#1D4ED8',
                              background: '#EFF6FF',
                              border: '1px solid #BFDBFE',
                              borderRadius: '4px',
                              cursor: 'pointer',
                            }}
                          >
                            <span>📊</span>
                            <span>Visualize this</span>
                          </button>
                        )}
                      </div>
                      <MarkdownRenderer content={m.content} style={{ fontSize: '12.5px', color: '#1F2937', lineHeight: 1.6 }} />
                      {m.role === 'assistant' && m.visualization && (
                        <VisualizationRenderer spec={m.visualization} />
                      )}
                      {m.role === 'assistant' && m.sources && m.sources.length > 0 && (
                        <div style={{ marginTop: '6px', paddingTop: '6px', borderTop: '1px solid #F1F5F9', display: 'flex', flexDirection: 'column', gap: '3px' }}>
                          <span style={{ fontSize: '9.5px', fontWeight: 600, color: '#6B7280', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                            Sources
                          </span>
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                            {m.sources.map((s, sIdx) => {
                              const isThread = s.type === 'thread' || s.title?.startsWith('Thread:')
                              return (
                                <span
                                  key={sIdx}
                                  title={s.excerpt || s.title}
                                  style={{
                                    display: 'inline-flex',
                                    alignItems: 'center',
                                    gap: '3px',
                                    fontSize: '10px',
                                    padding: '2px 6px',
                                    borderRadius: '4px',
                                    background: isThread ? '#EFF6FF' : '#F1F5F9',
                                    color: isThread ? '#1D4ED8' : '#374151',
                                    border: isThread ? '1px solid #BFDBFE' : '1px solid #E2E8F0',
                                  }}
                                >
                                  {isThread ? '💬 ' : '📄 '}
                                  {s.title}
                                </span>
                              )
                            })}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                  {isSending && (
                    <div style={{ fontSize: '11px', color: '#6B7280', fontStyle: 'italic', padding: '4px 8px' }}>
                      Retrieving department context & generating answer…
                    </div>
                  )}
                  <div ref={messagesEndRef} />
                </div>
              )}
            </div>
          )}

          {/* TAB 2: DOCUMENTS VIEW */}
          {(activeWsTab === 'documents' || currentNavView === 'documents') && (
            <div style={{ maxWidth: '900px', margin: '0 auto' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
                <div>
                  <h2 style={{ fontSize: '18px', fontWeight: 700, margin: 0, color: '#111827' }}>
                    {activeWorkspace ? `${activeWorkspace.name} Documents` : 'All Authorized Documents'}
                  </h2>
                  <div style={{ fontSize: '12px', color: '#6B7280', marginTop: '2px' }}>
                    Department Scope: <strong>{activeDeptCode || 'ALL'}</strong> · Permission: <strong>{currentPermissions.join(', ')}</strong>
                  </div>
                </div>

                {canUpload && (
                  <form onSubmit={handleUploadDocument} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <input
                      type="file"
                      id="doc-upload"
                      style={{ display: 'none' }}
                      onChange={(e) => {
                        if (e.target.files && e.target.files[0]) {
                          setUploadFile(e.target.files[0])
                        }
                      }}
                    />
                    <label
                      htmlFor="doc-upload"
                      style={{
                        padding: '6px 12px',
                        borderRadius: '6px',
                        background: '#FFFFFF',
                        border: '1px solid #D1D5DB',
                        fontSize: '11.5px',
                        cursor: 'pointer',
                        color: '#374151',
                      }}
                    >
                      {uploadFile ? uploadFile.name : 'Choose File'}
                    </label>
                    <button
                      type="submit"
                      disabled={!uploadFile || isUploading}
                      style={{
                        padding: '6px 14px',
                        borderRadius: '6px',
                        background: uploadFile ? '#1E293B' : '#94A3B8',
                        color: '#FFFFFF',
                        border: 'none',
                        fontSize: '11.5px',
                        cursor: uploadFile && !isUploading ? 'pointer' : 'default',
                      }}
                    >
                      {isUploading ? 'Uploading…' : '+ Upload Document'}
                    </button>
                  </form>
                )}
              </div>

              {uploadSuccess && <div style={{ fontSize: '12px', color: '#166534', marginBottom: '10px' }}>{uploadSuccess}</div>}
              {uploadError && <div style={{ fontSize: '12px', color: '#991B1B', marginBottom: '10px' }}>{uploadError}</div>}

              {/* Documents Table */}
              <div style={{ border: '1px solid #E5E7EB', borderRadius: '8px', overflow: 'hidden' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '12px' }}>
                  <thead>
                    <tr style={{ background: '#F8FAFC', borderBottom: '1px solid #E2E8F0' }}>
                      <th style={{ padding: '8px 12px', color: '#64748B', fontWeight: 600 }}>File Name</th>
                      <th style={{ padding: '8px 12px', color: '#64748B', fontWeight: 600 }}>Department</th>
                      <th style={{ padding: '8px 12px', color: '#64748B', fontWeight: 600 }}>Uploaded By</th>
                      <th style={{ padding: '8px 12px', color: '#64748B', fontWeight: 600 }}>Date</th>
                      <th style={{ padding: '8px 12px', color: '#64748B', fontWeight: 600 }}>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {loadingDocs ? (
                      <tr>
                        <td colSpan={5} style={{ padding: '24px', textAlign: 'center', color: '#9CA3AF' }}>Loading documents…</td>
                      </tr>
                    ) : workspaceDocs.length === 0 ? (
                      <tr>
                        <td colSpan={5} style={{ padding: '24px', textAlign: 'center', color: '#9CA3AF' }}>
                          No documents uploaded to this department yet.
                        </td>
                      </tr>
                    ) : (
                      workspaceDocs.map((doc) => (
                        <tr key={doc.id} style={{ borderBottom: '1px solid #F1F5F9' }}>
                          <td style={{ padding: '8px 12px', fontWeight: 500, color: '#1E293B' }}>
                            <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
                              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#64748B" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                                <polyline points="14 2 14 8 20 8" />
                              </svg>
                              {doc.file_name}
                            </span>
                          </td>
                          <td style={{ padding: '8px 12px', color: '#475569' }}>{doc.department_name || doc.department_code || 'Unassigned'}</td>
                          <td style={{ padding: '8px 12px', color: '#64748B' }}>{doc.uploaded_by_email || 'System'}</td>
                          <td style={{ padding: '8px 12px', color: '#64748B' }}>{doc.created_at ? new Date(doc.created_at).toLocaleDateString() : '-'}</td>
                          <td style={{ padding: '8px 12px' }}>
                            <span style={{ fontSize: '10px', padding: '2px 6px', borderRadius: '4px', background: '#ECFDF5', color: '#047857', border: '1px solid #A7F3D0' }}>
                              Indexed
                            </span>
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* TAB 3: KNOWLEDGE VIEW */}
          {(activeWsTab === 'knowledge' || currentNavView === 'knowledge') && (
            <div style={{ maxWidth: '820px', margin: '0 auto' }}>
              <h2 style={{ fontSize: '18px', fontWeight: 700, margin: '0 0 8px 0', color: '#111827' }}>
                {activeWorkspace ? `${activeWorkspace.name} Knowledge Base` : 'Knowledge Explorer'}
              </h2>
              <p style={{ fontSize: '12px', color: '#64748B', marginBottom: '16px' }}>
                Extracted entities, facts, and relationships scoped strictly to {activeWorkspace?.name || 'your authorized departments'}.
              </p>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '12px' }}>
                <div style={{ padding: '14px', border: '1px solid #E2E8F0', borderRadius: '8px', background: '#F8FAFC' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 600, fontSize: '13px', color: '#1E293B' }}>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#475569" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <rect x="4" y="2" width="16" height="20" rx="2" ry="2" />
                      <line x1="9" y1="22" x2="9" y2="22.01" />
                      <line x1="15" y1="22" x2="15" y2="22.01" />
                    </svg>
                    <span>Companies & Entities</span>
                  </div>
                  <div style={{ fontSize: '11px', color: '#64748B', marginTop: '4px' }}>Department-scoped extracted organization entities.</div>
                </div>
                <div style={{ padding: '14px', border: '1px solid #E2E8F0', borderRadius: '8px', background: '#F8FAFC' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 600, fontSize: '13px', color: '#1E293B' }}>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#475569" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
                      <circle cx="9" cy="7" r="4" />
                      <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
                      <path d="M16 3.13a4 4 0 0 1 0 7.75" />
                    </svg>
                    <span>People & Roles</span>
                  </div>
                  <div style={{ fontSize: '11px', color: '#64748B', marginTop: '4px' }}>Extracted staff, stakeholders, and department owners.</div>
                </div>
                <div style={{ padding: '14px', border: '1px solid #E2E8F0', borderRadius: '8px', background: '#F8FAFC' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 600, fontSize: '13px', color: '#1E293B' }}>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#475569" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <line x1="18" y1="20" x2="18" y2="10" />
                      <line x1="12" y1="20" x2="12" y2="4" />
                      <line x1="6" y1="20" x2="6" y2="14" />
                    </svg>
                    <span>Extracted Facts</span>
                  </div>
                  <div style={{ fontSize: '11px', color: '#64748B', marginTop: '4px' }}>Knowledge triples and verified attributes for citations.</div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 4: MEMBERS & ROLES */}
          {activeWsTab === 'members' && activeWorkspace && (
            <div style={{ maxWidth: '820px', margin: '0 auto' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
                <div>
                  <h2 style={{ fontSize: '18px', fontWeight: 700, margin: 0, color: '#111827' }}>
                    {activeWorkspace.name} Members & Permissions
                  </h2>
                  <div style={{ fontSize: '12px', color: '#64748B', marginTop: '2px' }}>
                    Users authorized to access {activeWorkspace.name} workspace.
                  </div>
                </div>
                {currentUser?.role === 'ADMIN' && (
                  <button
                    onClick={() => setMembersModalWs(activeWorkspace)}
                    style={{ padding: '6px 12px', borderRadius: '6px', background: '#1E293B', color: '#FFFFFF', border: 'none', fontSize: '11.5px', cursor: 'pointer' }}
                  >
                    Manage Members
                  </button>
                )}
              </div>

              <div style={{ padding: '16px', border: '1px solid #E2E8F0', borderRadius: '8px', background: '#F8FAFC' }}>
                <div style={{ fontSize: '12px', color: '#334155' }}>
                  Current user: <strong>{currentUser?.email}</strong> ({currentUser?.role})
                </div>
                <div style={{ fontSize: '11.5px', color: '#64748B', marginTop: '4px' }}>
                  Authorized Permissions in this Department: <strong>{currentPermissions.join(', ') || 'VIEW'}</strong>
                </div>
              </div>
            </div>
          )}

          {/* TAB 5: PROVIDER CONFIGURATION */}
          {activeWsTab === 'provider' && activeWorkspace && (
            <div style={{ maxWidth: '820px', margin: '0 auto' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
                <div>
                  <h2 style={{ fontSize: '18px', fontWeight: 700, margin: 0, color: '#111827' }}>
                    {activeWorkspace.name} Provider Configuration
                  </h2>
                  <div style={{ fontSize: '12px', color: '#64748B', marginTop: '2px' }}>
                    Configure specific LLM, embedding, and vector database parameters for this workspace.
                  </div>
                </div>
                <button
                  onClick={() => setSettingsModalWs(activeWorkspace)}
                  style={{ padding: '6px 12px', borderRadius: '6px', background: '#1E293B', color: '#FFFFFF', border: 'none', fontSize: '11.5px', cursor: 'pointer' }}
                >
                  Edit Provider Settings
                </button>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px' }}>
                <div style={{ padding: '14px', border: '1px solid #E2E8F0', borderRadius: '8px', background: '#F8FAFC' }}>
                  <div style={{ fontSize: '11px', color: '#64748B' }}>LLM Provider</div>
                  <div style={{ fontWeight: 600, fontSize: '13px', color: '#1E293B', marginTop: '2px' }}>Groq / OpenAI</div>
                </div>
                <div style={{ padding: '14px', border: '1px solid #E2E8F0', borderRadius: '8px', background: '#F8FAFC' }}>
                  <div style={{ fontSize: '11px', color: '#64748B' }}>Active Model</div>
                  <div style={{ fontWeight: 600, fontSize: '13px', color: '#1E293B', marginTop: '2px' }}>qwen/qwen3.5-9b</div>
                </div>
                <div style={{ padding: '14px', border: '1px solid #E2E8F0', borderRadius: '8px', background: '#F8FAFC' }}>
                  <div style={{ fontSize: '11px', color: '#64748B' }}>Embedding Provider</div>
                  <div style={{ fontWeight: 600, fontSize: '13px', color: '#1E293B', marginTop: '2px' }}>sentence_transformers</div>
                </div>
                <div style={{ padding: '14px', border: '1px solid #E2E8F0', borderRadius: '8px', background: '#F8FAFC' }}>
                  <div style={{ fontSize: '11px', color: '#64748B' }}>Vector Store</div>
                  <div style={{ fontWeight: 600, fontSize: '13px', color: '#1E293B', marginTop: '2px' }}>PostgreSQL + pgvector</div>
                </div>
              </div>
            </div>
          )}

          {/* GLOBAL VIEW: DEPARTMENTS (Admin) */}
          {currentNavView === 'departments' && (
            <div style={{ maxWidth: '820px', margin: '0 auto' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
                <div>
                  <h2 style={{ fontSize: '18px', fontWeight: 700, margin: 0, color: '#111827' }}>Department Management</h2>
                  <div style={{ fontSize: '12px', color: '#64748B', marginTop: '2px' }}>
                    Create and manage active departments. Each department automatically generates a Workspace.
                  </div>
                </div>
                {currentUser?.role === 'ADMIN' && (
                  <button
                    onClick={() => setShowCreateDeptModal(true)}
                    style={{ padding: '6px 12px', borderRadius: '6px', background: '#1E293B', color: '#FFFFFF', border: 'none', fontSize: '11.5px', cursor: 'pointer' }}
                  >
                    + Create Department
                  </button>
                )}
              </div>

              <div style={{ border: '1px solid #E5E7EB', borderRadius: '8px', overflow: 'hidden' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '12px' }}>
                  <thead>
                    <tr style={{ background: '#F8FAFC', borderBottom: '1px solid #E2E8F0' }}>
                      <th style={{ padding: '8px 12px', color: '#64748B', fontWeight: 600 }}>Name</th>
                      <th style={{ padding: '8px 12px', color: '#64748B', fontWeight: 600 }}>Code</th>
                      <th style={{ padding: '8px 12px', color: '#64748B', fontWeight: 600 }}>Status</th>
                      <th style={{ padding: '8px 12px', color: '#64748B', fontWeight: 600 }}>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {departments.map((d) => (
                      <tr key={d.id} style={{ borderBottom: '1px solid #F1F5F9' }}>
                        <td style={{ padding: '8px 12px', fontWeight: 500, color: '#1E293B' }}>{d.name}</td>
                        <td style={{ padding: '8px 12px', fontFamily: 'monospace', color: '#475569' }}>{d.code}</td>
                        <td style={{ padding: '8px 12px' }}>
                          <span style={{ fontSize: '10.5px', padding: '2px 6px', borderRadius: '4px', background: d.is_active ? '#ECFDF5' : '#FEF2F2', color: d.is_active ? '#047857' : '#991B1B' }}>
                            {d.is_active ? 'Active' : 'Inactive'}
                          </span>
                        </td>
                        <td style={{ padding: '8px 12px' }}>
                          {currentUser?.role === 'ADMIN' && (
                            <button
                              onClick={async () => {
                                try {
                                  await updateDepartmentStatus(d.code, !d.is_active, token)
                                  showToast('success', `Department status updated.`)
                                  loadDepartmentsAndWorkspaces()
                                } catch (err: any) {
                                  showToast('error', err.message || 'Status update failed.')
                                }
                              }}
                              style={{ fontSize: '11px', padding: '2px 6px', background: '#F1F5F9', border: '1px solid #E2E8F0', borderRadius: '4px', cursor: 'pointer' }}
                            >
                              {d.is_active ? 'Deactivate' : 'Activate'}
                            </button>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* GLOBAL VIEW: REQUESTS */}
          {currentNavView === 'requests' && (
            <div style={{ maxWidth: '820px', margin: '0 auto' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
                <div>
                  <h2 style={{ fontSize: '18px', fontWeight: 700, margin: 0, color: '#111827' }}>Access Requests</h2>
                  <div style={{ fontSize: '12px', color: '#64748B', marginTop: '2px' }}>
                    Request permissions (VIEW, UPLOAD, EDIT, DELETE) for other departments.
                  </div>
                </div>
                <button
                  onClick={() => setShowAccessReqModal(true)}
                  style={{ padding: '6px 12px', borderRadius: '6px', background: '#1E293B', color: '#FFFFFF', border: 'none', fontSize: '11.5px', cursor: 'pointer' }}
                >
                  + Request Department Access
                </button>
              </div>

              <div style={{ border: '1px solid #E5E7EB', borderRadius: '8px', overflow: 'hidden' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '12px' }}>
                  <thead>
                    <tr style={{ background: '#F8FAFC', borderBottom: '1px solid #E2E8F0' }}>
                      <th style={{ padding: '8px 12px', color: '#64748B', fontWeight: 600 }}>Department</th>
                      <th style={{ padding: '8px 12px', color: '#64748B', fontWeight: 600 }}>Permissions</th>
                      <th style={{ padding: '8px 12px', color: '#64748B', fontWeight: 600 }}>Status</th>
                      <th style={{ padding: '8px 12px', color: '#64748B', fontWeight: 600 }}>Reason</th>
                    </tr>
                  </thead>
                  <tbody>
                    {accessRequests.length === 0 ? (
                      <tr>
                        <td colSpan={4} style={{ padding: '24px', textAlign: 'center', color: '#9CA3AF' }}>No access requests submitted.</td>
                      </tr>
                    ) : (
                      accessRequests.map((req) => (
                        <tr key={req.id} style={{ borderBottom: '1px solid #F1F5F9' }}>
                          <td style={{ padding: '8px 12px', fontWeight: 500, color: '#1E293B' }}>{req.department_name || req.department_code}</td>
                          <td style={{ padding: '8px 12px', color: '#475569' }}>{(req.requested_permissions || (req as any).permissions)?.join(', ')}</td>
                          <td style={{ padding: '8px 12px' }}>
                            <span style={{ fontSize: '10.5px', padding: '2px 6px', borderRadius: '4px', background: req.status === 'APPROVED' ? '#ECFDF5' : req.status === 'PENDING' ? '#FFFBEB' : '#FEF2F2', color: req.status === 'APPROVED' ? '#047857' : req.status === 'PENDING' ? '#B45309' : '#991B1B' }}>
                              {req.status}
                            </span>
                          </td>
                          <td style={{ padding: '8px 12px', color: '#64748B' }}>{req.reason || '-'}</td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>

        {/* ========================================================================= */}
        {/* MESSAGE COMPOSER (Floating / Centered Bottom)                             */}
        {/* ========================================================================= */}
        {activeWsTab === 'threads' && currentNavView === 'chat' && (
          <div
            style={{
              position: 'absolute',
              bottom: '16px',
              left: '50%',
              transform: 'translateX(-50%)',
              width: '400px',
              maxWidth: 'calc(100% - 32px)',
              background: '#FFFFFF',
              border: '1px solid #E3E3E3',
              borderRadius: '13px',
              boxShadow: '0 4px 18px rgba(0, 0, 0, 0.06)',
              padding: '7px 10px 6px 10px',
              boxSizing: 'border-box',
              display: 'flex',
              flexDirection: 'column',
              gap: '3px',
              zIndex: 10,
            }}
          >
            <textarea
              value={messageInput}
              onChange={(e) => setMessageInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  handleSendMessage()
                }
              }}
              placeholder={
                activeWorkspace
                  ? `Send a message in ${activeWorkspace.name}…`
                  : 'Ask a cross-department question across all departments…'
              }
              rows={1}
              style={{
                width: '100%',
                border: 'none',
                outline: 'none',
                resize: 'none',
                fontSize: '11.5px',
                fontFamily: 'inherit',
                color: '#252525',
                background: 'transparent',
                padding: '2px',
                minHeight: '26px',
                maxHeight: '80px',
                boxSizing: 'border-box',
              }}
            />

            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '2px' }}>
              {/* + Tools Button */}
              <div style={{ position: 'relative' }}>
                <button
                  type="button"
                  onClick={() => setToolsOpen(!toolsOpen)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '3px',
                    fontSize: '10.5px',
                    fontWeight: 500,
                    color: '#6B7280',
                    background: '#F7F7F8',
                    border: '1px solid #E5E7EB',
                    borderRadius: '5px',
                    padding: '2px 6px',
                    cursor: 'pointer',
                  }}
                >
                  <span style={{ fontSize: '11px', fontWeight: 600 }}>+</span>
                  <span>Tools</span>
                </button>

                {toolsOpen && (
                  <div
                    style={{
                      position: 'absolute',
                      bottom: '28px',
                      left: 0,
                      background: '#FFFFFF',
                      border: '1px solid #E5E7EB',
                      borderRadius: '8px',
                      boxShadow: '0 4px 14px rgba(0,0,0,0.1)',
                      padding: '6px',
                      width: '150px',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '4px',
                      zIndex: 20,
                    }}
                  >
                    <div
                      onClick={() => {
                        setToolsOpen(false)
                        setActiveWsTab('documents')
                      }}
                      style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', padding: '5px 6px', borderRadius: '4px', cursor: 'pointer', color: '#374151' }}
                      onMouseEnter={(e) => (e.currentTarget.style.background = '#F3F4F6')}
                      onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                    >
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#64748B" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                        <polyline points="14 2 14 8 20 8" />
                      </svg>
                      <span>Search Documents</span>
                    </div>
                    <div
                      onClick={() => {
                        setToolsOpen(false)
                        setActiveWsTab('knowledge')
                      }}
                      style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', padding: '5px 6px', borderRadius: '4px', cursor: 'pointer', color: '#374151' }}
                      onMouseEnter={(e) => (e.currentTarget.style.background = '#F3F4F6')}
                      onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                    >
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#64748B" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="12" cy="12" r="3" />
                        <circle cx="6" cy="6" r="2" />
                        <circle cx="18" cy="6" r="2" />
                        <circle cx="6" cy="18" r="2" />
                        <circle cx="18" cy="18" r="2" />
                      </svg>
                      <span>Query Knowledge Base</span>
                    </div>
                  </div>
                )}
              </div>

              {/* Right Side: Microphone & Send Button */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                <button
                  type="button"
                  onClick={() => setIsRecording(!isRecording)}
                  title={isRecording ? 'Listening…' : 'Voice input'}
                  style={{
                    background: isRecording ? '#FEE2E2' : 'transparent',
                    border: 'none',
                    cursor: 'pointer',
                    padding: '3px',
                    borderRadius: '50%',
                    color: isRecording ? '#DC2626' : '#6B7280',
                    display: 'flex',
                  }}
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
                    <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                    <line x1="12" y1="19" x2="12" y2="23" />
                    <line x1="8" y1="23" x2="16" y2="23" />
                  </svg>
                </button>

                <button
                  type="button"
                  onClick={handleSendMessage}
                  disabled={isSending || !messageInput.trim()}
                  title="Send message"
                  style={{
                    width: '23px',
                    height: '23px',
                    borderRadius: '50%',
                    background: messageInput.trim() ? '#1E293B' : '#64748B',
                    border: 'none',
                    color: '#FFFFFF',
                    cursor: messageInput.trim() ? 'pointer' : 'default',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    padding: 0,
                    opacity: messageInput.trim() ? 1 : 0.8,
                  }}
                >
                  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="12" y1="19" x2="12" y2="5" />
                    <polyline points="5 12 12 5 19 12" />
                  </svg>
                </button>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* ========================================================================= */}
      {/* MODAL 1: CREATE DEPARTMENT (Admin Only)                                  */}
      {/* ========================================================================= */}
      {showCreateDeptModal && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0,0,0,0.4)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 9000,
          }}
          onClick={() => setShowCreateDeptModal(false)}
        >
          <div
            style={{
              background: '#FFFFFF',
              borderRadius: '12px',
              padding: '20px',
              width: '360px',
              boxShadow: '0 10px 30px rgba(0,0,0,0.15)',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <h3 style={{ margin: '0 0 14px 0', fontSize: '15px', color: '#111827' }}>+ Create Department</h3>
            {deptModalError && <div style={{ fontSize: '11px', color: '#DC2626', marginBottom: '10px' }}>{deptModalError}</div>}
            <form onSubmit={handleCreateDepartment}>
              <div style={{ marginBottom: '12px' }}>
                <label style={{ display: 'block', fontSize: '11px', color: '#6B7280', marginBottom: '4px' }}>Department Name</label>
                <input
                  type="text"
                  placeholder="e.g. Legal"
                  value={newDeptName}
                  onChange={(e) => setNewDeptName(e.target.value)}
                  style={{ width: '100%', boxSizing: 'border-box', padding: '6px 10px', borderRadius: '6px', border: '1px solid #D1D5DB', fontSize: '12px' }}
                />
              </div>
              <div style={{ marginBottom: '16px' }}>
                <label style={{ display: 'block', fontSize: '11px', color: '#6B7280', marginBottom: '4px' }}>Department Code</label>
                <input
                  type="text"
                  placeholder="e.g. LEGAL"
                  value={newDeptCode}
                  onChange={(e) => setNewDeptCode(e.target.value)}
                  style={{ width: '100%', boxSizing: 'border-box', padding: '6px 10px', borderRadius: '6px', border: '1px solid #D1D5DB', fontSize: '12px', textTransform: 'uppercase' }}
                />
              </div>
              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
                <button
                  type="button"
                  onClick={() => setShowCreateDeptModal(false)}
                  style={{ padding: '6px 12px', borderRadius: '6px', background: '#F3F4F6', border: '1px solid #E5E7EB', fontSize: '11.5px', cursor: 'pointer' }}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={creatingDept}
                  style={{ padding: '6px 14px', borderRadius: '6px', background: '#1E293B', color: '#FFFFFF', border: 'none', fontSize: '11.5px', cursor: 'pointer' }}
                >
                  {creatingDept ? 'Creating…' : 'Create'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* MODAL 2: ACCESS REQUEST                                                   */}
      {/* ========================================================================= */}
      {showAccessReqModal && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0,0,0,0.4)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 9000,
          }}
          onClick={() => setShowAccessReqModal(false)}
        >
          <div
            style={{
              background: '#FFFFFF',
              borderRadius: '12px',
              padding: '20px',
              width: '380px',
              boxShadow: '0 10px 30px rgba(0,0,0,0.15)',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <h3 style={{ margin: '0 0 14px 0', fontSize: '15px', color: '#111827' }}>Request Department Access</h3>
            <form onSubmit={handleSubmitAccessRequest}>
              <div style={{ marginBottom: '12px' }}>
                <label style={{ display: 'block', fontSize: '11px', color: '#6B7280', marginBottom: '4px' }}>Select Department</label>
                <select
                  value={accessReqDept}
                  onChange={(e) => setAccessReqDept(e.target.value)}
                  style={{ width: '100%', padding: '6px 10px', borderRadius: '6px', border: '1px solid #D1D5DB', fontSize: '12px' }}
                >
                  <option value="">Choose department…</option>
                  {departments.map((d) => (
                    <option key={d.code} value={d.code}>
                      {d.name} ({d.code})
                    </option>
                  ))}
                </select>
              </div>

              <div style={{ marginBottom: '12px' }}>
                <label style={{ display: 'block', fontSize: '11px', color: '#6B7280', marginBottom: '4px' }}>Requested Permissions</label>
                <div style={{ display: 'flex', gap: '8px' }}>
                  {(['VIEW', 'UPLOAD', 'EDIT', 'DELETE'] as PermissionCode[]).map((p) => (
                    <label key={p} style={{ fontSize: '11px', display: 'flex', alignItems: 'center', gap: '3px' }}>
                      <input
                        type="checkbox"
                        checked={accessReqPerms.includes(p)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setAccessReqPerms([...accessReqPerms, p])
                          } else {
                            setAccessReqPerms(accessReqPerms.filter((x) => x !== p))
                          }
                        }}
                      />
                      {p}
                    </label>
                  ))}
                </div>
              </div>

              <div style={{ marginBottom: '16px' }}>
                <label style={{ display: 'block', fontSize: '11px', color: '#6B7280', marginBottom: '4px' }}>Reason</label>
                <textarea
                  rows={2}
                  placeholder="Why do you need access to this department?"
                  value={accessReqReason}
                  onChange={(e) => setAccessReqReason(e.target.value)}
                  style={{ width: '100%', boxSizing: 'border-box', padding: '6px 10px', borderRadius: '6px', border: '1px solid #D1D5DB', fontSize: '12px' }}
                />
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
                <button
                  type="button"
                  onClick={() => setShowAccessReqModal(false)}
                  style={{ padding: '6px 12px', borderRadius: '6px', background: '#F3F4F6', border: '1px solid #E5E7EB', fontSize: '11.5px', cursor: 'pointer' }}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={submittingAccessReq}
                  style={{ padding: '6px 14px', borderRadius: '6px', background: '#1E293B', color: '#FFFFFF', border: 'none', fontSize: '11.5px', cursor: 'pointer' }}
                >
                  {submittingAccessReq ? 'Submitting…' : 'Submit Request'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* MODAL 3: WORKSPACE PROVIDER SETTINGS                                      */}
      {/* ========================================================================= */}
      {settingsModalWs && (
        <WorkspaceProviderSettingsModal
          workspace={settingsModalWs}
          token={token}
          isOwnerOrAdmin={currentUser?.role === 'ADMIN'}
          onClose={() => setSettingsModalWs(null)}
          onToast={showToast}
        />
      )}

      {/* ========================================================================= */}
      {/* MODAL 4: WORKSPACE MEMBERS                                                */}
      {/* ========================================================================= */}
      {membersModalWs && (
        <WorkspaceMembersModal
          workspace={membersModalWs}
          token={token}
          currentUserEmail={currentUser?.email}
          isOwnerOrAdmin={currentUser?.role === 'ADMIN'}
          onClose={() => setMembersModalWs(null)}
          onToast={showToast}
        />
      )}
    </div>
  )
}
