import { useEffect, useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  adminApi,
  UserAccessSummary,
  DeptAccessSummary,
  DepartmentResponse,
  PermissionResponse,
} from '../api/admin'
import { departmentApi } from '../api/departments'
import {
  accessRequestApi,
  AccessRequestResponse,
  PermissionCode,
  AccessRequestStatus,
} from '../api/accessRequests'
import { userApi } from '../api/user'
import Sidebar, { NavItem } from '../components/Sidebar'
import RAG from '../components/RAG'
import {
  ConversationItem,
  listConversations,
  deleteConversation,
} from '../api/chat'
import { documentApi, DocumentListItem, DocumentAccessListResponse, DocUserAccessItem } from '../api/documents'
import AnythingLLMApp from '../components/AnythingLLMApp'

/* ------------------------------------------------------------------ */
/*  Types & Constants                                                  */
/* ------------------------------------------------------------------ */

const ALL_PERMS = ['VIEW', 'UPLOAD', 'EDIT', 'DELETE'] as const
type PermCode = (typeof ALL_PERMS)[number]

const PRESETS: Record<string, PermCode[]> = {
  'View Only': ['VIEW'],
  Contributor: ['VIEW', 'UPLOAD', 'EDIT'],
  'Full Access': ['VIEW', 'UPLOAD', 'EDIT', 'DELETE'],
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

function getDisplayName(email: string) {
  const local = email.split('@')[0]
  return local.charAt(0).toUpperCase() + local.slice(1)
}

function accessLabel(user: UserAccessSummary): string {
  if (user.role === 'ADMIN') return 'Global Access'
  if (user.departments.length === 0) return 'No Access'
  return user.departments
    .map((d) => {
      const perms = d.permissions.length > 0 ? d.permissions.join(', ') : 'No Perms'
      return `${d.name || d.code} · ${perms}`
    })
    .join('  |  ')
}

function accessBadgeColor(user: UserAccessSummary) {
  if (user.role === 'ADMIN') return { bg: '#e0f2f1', color: '#00695c' }
  if (user.departments.length === 0) return { bg: '#fff3e0', color: '#e65100' }
  return { bg: '#e3f2fd', color: '#1565c0' }
}

function accessRequestStatusBadge(status: AccessRequestStatus) {
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

function matchesPreset(perms: string[]): string | null {
  for (const [name, codes] of Object.entries(PRESETS)) {
    if (codes.length === perms.length && codes.every((c) => perms.includes(c))) return name
  }
  return null
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export default function AdminDashboard({
  token,
  onLogout,
}: {
  token: string
  onLogout: () => void
}) {
  return <AnythingLLMApp token={token} onLogout={onLogout} />
  const [users, setUsers] = useState<UserAccessSummary[]>([])
  const [filteredUsers, setFilteredUsers] = useState<UserAccessSummary[]>([])

  // Filters
  const [search, setSearch] = useState('')
  const [roleFilter, setRoleFilter] = useState<'ALL' | 'ADMIN' | 'USER'>('ALL')
  const [accessFilter, setAccessFilter] = useState<'ALL' | 'GLOBAL' | 'SCOPED' | 'NONE'>('ALL')

  // Detail view
  const [selectedUserId, setSelectedUserId] = useState<string | null>(null)
  const [detailTab, setDetailTab] = useState<'profile' | 'access' | 'activity'>('access')
  const [detailDepts, setDetailDepts] = useState<DepartmentResponse[]>([])
  const [detailPerms, setDetailPerms] = useState<Record<string, PermissionResponse[]>>({})
  const [allDepartments, setAllDepartments] = useState<DepartmentResponse[]>([])

  // Create user modal
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [createEmail, setCreateEmail] = useState('')
  const [createPassword, setCreatePassword] = useState('')
  const [createRole, setCreateRole] = useState<'USER' | 'ADMIN'>('USER')
  const [createInitialDept, setCreateInitialDept] = useState('')
  const [createInitialPerms, setCreateInitialPerms] = useState<PermCode[]>(['VIEW'])
  const [createLoading, setCreateLoading] = useState(false)

  // Departments view state
  const [departments, setDepartments] = useState<DepartmentResponse[]>([])
  const [filteredDepartments, setFilteredDepartments] = useState<DepartmentResponse[]>([])
  const [deptSearch, setDeptSearch] = useState('')
  const [deptStatusFilter, setDeptStatusFilter] = useState<'ALL' | 'ACTIVE' | 'INACTIVE'>('ALL')
  const [showCreateDeptModal, setShowCreateDeptModal] = useState(false)
  const [createDeptCode, setCreateDeptCode] = useState('')
  const [createDeptName, setCreateDeptName] = useState('')
  const [createDeptLoading, setCreateDeptLoading] = useState(false)
  const [deactivateConfirmDept, setDeactivateConfirmDept] = useState<DepartmentResponse | null>(null)
  const [deptActionLoading, setDeptActionLoading] = useState(false)

  // Access Requests state
  const [accessRequests, setAccessRequests] = useState<AccessRequestResponse[]>([])
  const [filteredAccessRequests, setFilteredAccessRequests] = useState<AccessRequestResponse[]>([])
  const [requestsLoading, setRequestsLoading] = useState(false)
  const [requestSearch, setRequestSearch] = useState('')
  const [requestStatusFilter, setRequestStatusFilter] = useState<'ALL' | AccessRequestStatus>('ALL')
  const [pendingRequestsCount, setPendingRequestsCount] = useState(0)
  const [reviewModalRequest, setReviewModalRequest] = useState<AccessRequestResponse | null>(null)
  const [approvedPerms, setApprovedPerms] = useState<PermissionCode[]>([])
  const [reviewNote, setReviewNote] = useState('')
  const [reviewSubmitting, setReviewSubmitting] = useState(false)

  // Toasts
  const [toast, setToast] = useState<{ type: 'success' | 'error'; message: string } | null>(null)
  const [saving, setSaving] = useState(false)

  // Layout & Conversations
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false)
  const [activeView, setActiveView] = useState<NavItem>('chat')
  const [userName, setUserName] = useState('Admin')
  const [userEmail, setUserEmail] = useState('')
  const [allowedUploadDepts, setAllowedUploadDepts] = useState<{ code: string; name: string }[]>([])
  const [conversations, setConversations] = useState<ConversationItem[]>([])
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null)
  const [loadingConversations, setLoadingConversations] = useState(false)

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

  // Document Access Management State
  const [manageAccessDoc, setManageAccessDoc] = useState<DocumentListItem | null>(null)
  const [docAccessData, setDocAccessData] = useState<DocumentAccessListResponse | null>(null)
  const [docAccessLoading, setDocAccessLoading] = useState(false)
  const [pendingDocPerms, setPendingDocPerms] = useState<Record<string, string[]>>({})
  const [docAccessSavingId, setDocAccessSavingId] = useState<string | null>(null)
  const navigate = useNavigate()

  /* ---- Toast ---- */
  const showToast = useCallback((type: 'success' | 'error', message: string) => {
    setToast({ type, message })
    setTimeout(() => setToast(null), 4000)
  }, [])

  /* ---- Init User Profile ---- */
  useEffect(() => {
    ;(async () => {
      try {
        const me = await userApi.getMe(token)
        if (me.role !== 'ADMIN') {
          navigate('/app', { replace: true })
          return
        }
        setUserName(getDisplayName(me.email))
        setUserEmail(me.email)
      } catch {
        onLogout()
      }
    })()
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
    let isMounted = true
    loadConversations().then((items) => {
      if (!isMounted || !items || items.length === 0) return
      setActiveConversationId((prev) => prev || items[0].id)
    })
    return () => {
      isMounted = false
    }
  }, [loadConversations])


  const handleSelectConversation = (convId: string) => {
    setActiveConversationId(convId)
    setActiveView('chat')
    setSelectedUserId(null)
  }

  const handleNewChat = () => {
    setActiveConversationId(null)
    setActiveView('chat')
    setSelectedUserId(null)
  }

  const handleDeleteConversation = async (e: React.MouseEvent, convId: string) => {
    e.stopPropagation()
    try {
      await deleteConversation(convId, token)
      setConversations((prev) => {
        const remaining = prev.filter((c) => c.id !== convId)
        if (activeConversationId === convId) {
          setActiveConversationId(remaining.length > 0 ? remaining[0].id : null)
        }
        return remaining
      })
    } catch (err) {
      console.error('Failed to delete conversation:', err)
    }
  }


  /* ---- Load users ---- */
  const loadUsers = useCallback(async () => {
    try {
      const data = await adminApi.getUsersWithAccess(token)
      setUsers(data)
    } catch (e: any) {
      console.error(e)
    }
  }, [token])


  useEffect(() => {
    if (activeView === 'users') loadUsers()
  }, [activeView, loadUsers])

  /* ---- Load departments ---- */
  const loadAllDepartments = useCallback(async () => {
    try {
      const data = await departmentApi.getAllDepartments(token)
      setDepartments(data)
      setAllowedUploadDepts(
        data.filter((d) => d.is_active).map((d) => ({ code: d.code, name: d.name }))
      )
    } catch (e: any) {
      console.error(e)
      showToast('error', 'Failed to load departments: ' + e.message)
    }
  }, [token, showToast])

  useEffect(() => {
    loadAllDepartments()
  }, [loadAllDepartments])

  useEffect(() => {
    if (activeView === 'departments') {
      loadAllDepartments()
    }
  }, [activeView, loadAllDepartments])

  /* ---- Load access requests ---- */
  const loadAccessRequests = useCallback(async () => {
    setRequestsLoading(true)
    try {
      const data = await accessRequestApi.getAccessRequests(token)
      setAccessRequests(data)
      setPendingRequestsCount(data.filter((r) => r.status === 'PENDING').length)
    } catch (e: any) {
      console.error(e)
      showToast('error', 'Failed to load access requests: ' + e.message)
    } finally {
      setRequestsLoading(false)
    }
  }, [token, showToast])

  useEffect(() => {
    loadAccessRequests()
  }, [loadAccessRequests])

  useEffect(() => {
    if (activeView === 'requests') {
      loadAccessRequests()
    }
  }, [activeView, loadAccessRequests])

  /* ---- Access Request Filters ---- */
  useEffect(() => {
    let list = accessRequests
    if (requestSearch) {
      const q = requestSearch.toLowerCase()
      list = list.filter(
        (r) =>
          r.user_email.toLowerCase().includes(q) ||
          r.department_code.toLowerCase().includes(q) ||
          r.department_name.toLowerCase().includes(q) ||
          (r.reason && r.reason.toLowerCase().includes(q)) ||
          (r.reviewer_email && r.reviewer_email.toLowerCase().includes(q))
      )
    }
    if (requestStatusFilter !== 'ALL') {
      list = list.filter((r) => r.status === requestStatusFilter)
    }
    setFilteredAccessRequests(list)
  }, [accessRequests, requestSearch, requestStatusFilter])

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

  /* ---- Document Access Management Handlers ---- */
  const handleOpenManageAccess = async (doc: DocumentListItem) => {
    setManageAccessDoc(doc)
    setDocAccessLoading(true)
    setDocAccessData(null)
    setPendingDocPerms({})
    try {
      const data = await documentApi.getDocumentPermissions(doc.id, token)
      setDocAccessData(data)
      const initialPerms: Record<string, string[]> = {}
      data.users.forEach((u) => {
        initialPerms[u.user_id] = [...u.document_permissions]
      })
      setPendingDocPerms(initialPerms)
    } catch (e: any) {
      showToast('error', e.message || 'Failed to load document permissions')
    } finally {
      setDocAccessLoading(false)
    }
  }

  const handleToggleDocPerm = (userId: string, permCode: string) => {
    setPendingDocPerms((prev) => {
      const current = prev[userId] || []
      const has = current.includes(permCode)
      const next = has ? current.filter((p) => p !== permCode) : [...current, permCode]
      return { ...prev, [userId]: next }
    })
  }

  const handleSaveUserDocPerms = async (userId: string) => {
    if (!manageAccessDoc) return
    const targetPerms = pendingDocPerms[userId] || []
    setDocAccessSavingId(userId)
    try {
      await documentApi.setDocumentPermissions(manageAccessDoc.id, userId, targetPerms, token)
      showToast('success', 'Document permissions updated successfully')
      const data = await documentApi.getDocumentPermissions(manageAccessDoc.id, token)
      setDocAccessData(data)
      const updatedPerms: Record<string, string[]> = {}
      data.users.forEach((u) => {
        updatedPerms[u.user_id] = [...u.document_permissions]
      })
      setPendingDocPerms(updatedPerms)
    } catch (e: any) {
      showToast('error', e.message || 'Failed to update document permissions')
    } finally {
      setDocAccessSavingId(null)
    }
  }

  /* ---- Access Request Review Handlers ---- */
  function handleOpenReview(req: AccessRequestResponse) {
    setReviewModalRequest(req)
    const validRequested = ALL_PERMS.filter((p) => req.requested_permissions.includes(p))
    setApprovedPerms(validRequested.length > 0 ? validRequested : ['VIEW'])
    setReviewNote(req.review_note || '')
  }

  function handleToggleApprovedPerm(perm: PermissionCode) {
    if (approvedPerms.includes(perm)) {
      setApprovedPerms(approvedPerms.filter((p) => p !== perm))
    } else {
      setApprovedPerms([...approvedPerms, perm])
    }
  }

  async function handleApproveRequest() {
    if (!reviewModalRequest) return
    if (approvedPerms.length === 0) {
      showToast('error', 'At least one permission must be approved.')
      return
    }
    setReviewSubmitting(true)
    try {
      await accessRequestApi.approveAccessRequest(
        reviewModalRequest.id,
        approvedPerms,
        reviewNote.trim() || undefined,
        token
      )
      showToast('success', `Access request for ${reviewModalRequest.user_email} approved`)
      setReviewModalRequest(null)
      await loadAccessRequests()
      if (activeView === 'users') await loadUsers()
    } catch (e: any) {
      showToast('error', e.message)
    } finally {
      setReviewSubmitting(false)
    }
  }

  async function handleRejectRequest() {
    if (!reviewModalRequest) return
    setReviewSubmitting(true)
    try {
      await accessRequestApi.rejectAccessRequest(
        reviewModalRequest.id,
        reviewNote.trim() || undefined,
        token
      )
      showToast('success', `Access request for ${reviewModalRequest.user_email} rejected`)
      setReviewModalRequest(null)
      await loadAccessRequests()
    } catch (e: any) {
      showToast('error', e.message)
    } finally {
      setReviewSubmitting(false)
    }
  }

  /* ---- Department Filters ---- */
  useEffect(() => {
    let list = departments
    if (deptSearch) {
      const q = deptSearch.toLowerCase()
      list = list.filter(
        (d) => d.code.toLowerCase().includes(q) || d.name.toLowerCase().includes(q)
      )
    }
    if (deptStatusFilter === 'ACTIVE') {
      list = list.filter((d) => d.is_active)
    } else if (deptStatusFilter === 'INACTIVE') {
      list = list.filter((d) => !d.is_active)
    }
    setFilteredDepartments(list)
  }, [departments, deptSearch, deptStatusFilter])

  /* ---- Filters ---- */
  useEffect(() => {
    let list = users
    if (search) {
      const q = search.toLowerCase()
      list = list.filter(
        (u) => u.email.toLowerCase().includes(q) || getDisplayName(u.email).toLowerCase().includes(q),
      )
    }
    if (roleFilter !== 'ALL') list = list.filter((u) => u.role === roleFilter)
    if (accessFilter === 'GLOBAL') list = list.filter((u) => u.role === 'ADMIN')
    else if (accessFilter === 'SCOPED')
      list = list.filter((u) => u.role === 'USER' && u.departments.length > 0)
    else if (accessFilter === 'NONE')
      list = list.filter((u) => u.role === 'USER' && u.departments.length === 0)
    setFilteredUsers(list)
  }, [users, search, roleFilter, accessFilter])

  /* ---- Select user ---- */
  const selectUser = useCallback(
    async (userId: string) => {
      setSelectedUserId(userId)
      setDetailTab('access')
      setDetailDepts([])
      setDetailPerms({})
      try {
        const [depts, allDepts] = await Promise.all([
          adminApi.getUserDepartments(userId, token),
          adminApi.listDepartments(token),
        ])
        setDetailDepts(depts)
        setAllDepartments(allDepts)
        const permsMap: Record<string, PermissionResponse[]> = {}
        for (const d of depts) {
          permsMap[d.code] = await adminApi.getUserPermissions(userId, d.code, token)
        }
        setDetailPerms(permsMap)
      } catch (e: any) {
        showToast('error', 'Failed to load user access: ' + e.message)
      }
    },
    [token, showToast],
  )

  const selectedUser = users.find((u) => u.id === selectedUserId) ?? null

  /* ---- Actions ---- */
  async function createUser() {
    setCreateLoading(true)
    try {
      const payload: any = {
        email: createEmail.trim().toLowerCase(),
        password: createPassword,
        role: createRole,
      }
      if (createRole === 'USER' && createInitialDept) {
        payload.department_code = createInitialDept
        payload.permissions = createInitialPerms
      }
      await adminApi.createUser(payload, token)
      const accessNote =
        createRole === 'USER' && createInitialDept
          ? ` with initial access to ${createInitialDept} (${createInitialPerms.join(', ')})`
          : createRole === 'ADMIN'
          ? ' with Global Access'
          : ' with No Access'
      showToast('success', `User ${createEmail} created successfully${accessNote}`)
      setShowCreateModal(false)
      setCreateEmail('')
      setCreatePassword('')
      setCreateRole('USER')
      setCreateInitialDept('')
      setCreateInitialPerms(['VIEW'])
      loadUsers()
    } catch (e: any) {
      showToast('error', e.message)
    } finally {
      setCreateLoading(false)
    }
  }

  /* ---- Department Actions ---- */
  async function handleCreateDept() {
    const code = createDeptCode.trim().toUpperCase()
    const name = createDeptName.trim()
    if (!code || !name) {
      showToast('error', 'Both Department Code and Name are required.')
      return
    }
    setCreateDeptLoading(true)
    try {
      await departmentApi.createDepartment(code, name, token)
      showToast('success', `Department ${code} created successfully`)
      setShowCreateDeptModal(false)
      setCreateDeptCode('')
      setCreateDeptName('')
      await loadAllDepartments()
    } catch (e: any) {
      showToast('error', e.message)
    } finally {
      setCreateDeptLoading(false)
    }
  }

  async function handleActivateDept(dept: DepartmentResponse) {
    setDeptActionLoading(true)
    try {
      await departmentApi.updateDepartmentStatus(dept.code, true, token)
      showToast('success', `Department ${dept.code} activated`)
      await loadAllDepartments()
    } catch (e: any) {
      showToast('error', e.message)
    } finally {
      setDeptActionLoading(false)
    }
  }

  async function handleDeactivateDept() {
    if (!deactivateConfirmDept) return
    setDeptActionLoading(true)
    try {
      await departmentApi.updateDepartmentStatus(deactivateConfirmDept.code, false, token)
      showToast('success', `Department ${deactivateConfirmDept.code} deactivated`)
      setDeactivateConfirmDept(null)
      await loadAllDepartments()
    } catch (e: any) {
      showToast('error', e.message)
    } finally {
      setDeptActionLoading(false)
    }
  }

  async function toggleStatus() {
    if (!selectedUser) return
    setSaving(true)
    try {
      await adminApi.updateUserStatus(selectedUser.id, !selectedUser.is_active, token)
      showToast('success', `User ${selectedUser.is_active ? 'deactivated' : 'activated'}`)
      await loadUsers()
    } catch (e: any) {
      showToast('error', e.message)
    } finally {
      setSaving(false)
    }
  }

  async function assignDept(deptCode: string) {
    if (!selectedUser) return
    setSaving(true)
    try {
      await adminApi.assignUserDepartment(selectedUser.id, deptCode, token)
      showToast('success', `Department ${deptCode} assigned`)
      await selectUser(selectedUser.id)
      await loadUsers()
    } catch (e: any) {
      showToast('error', e.message)
    } finally {
      setSaving(false)
    }
  }

  async function removeDept(deptCode: string) {
    if (!selectedUser) return
    setSaving(true)
    try {
      await adminApi.revokeUserDepartment(selectedUser.id, deptCode, token)
      showToast('success', `Department ${deptCode} removed`)
      await selectUser(selectedUser.id)
      await loadUsers()
    } catch (e: any) {
      showToast('error', e.message)
    } finally {
      setSaving(false)
    }
  }

  async function togglePerm(deptCode: string, permCode: string, hasPerm: boolean) {
    if (!selectedUser) return
    setSaving(true)
    try {
      if (hasPerm) {
        await adminApi.revokeUserPermission(selectedUser.id, deptCode, permCode, token)
      } else {
        await adminApi.grantUserPermission(selectedUser.id, deptCode, permCode, token)
      }
      await selectUser(selectedUser.id)
      await loadUsers()
    } catch (e: any) {
      showToast('error', e.message)
    } finally {
      setSaving(false)
    }
  }

  async function applyPreset(deptCode: string, presetName: string) {
    if (!selectedUser) return
    const codes = PRESETS[presetName]
    if (!codes) return
    setSaving(true)
    try {
      // Remove current perms that aren't in preset
      const currentPerms = (detailPerms[deptCode] || []).map((p) => p.code)
      for (const p of currentPerms) {
        if (!codes.includes(p as PermCode)) {
          await adminApi.revokeUserPermission(selectedUser.id, deptCode, p, token)
        }
      }
      // Grant missing
      for (const p of codes) {
        if (!currentPerms.includes(p)) {
          await adminApi.grantUserPermission(selectedUser.id, deptCode, p, token)
        }
      }
      showToast('success', `${presetName} preset applied to ${deptCode}`)
      await selectUser(selectedUser.id)
      await loadUsers()
    } catch (e: any) {
      showToast('error', e.message)
    } finally {
      setSaving(false)
    }
  }

  /* ================================================================ */
  /*  RENDER                                                           */
  /* ================================================================ */

  const unassignedDepts = allDepartments.filter(
    (d) => !detailDepts.some((ud) => ud.code === d.code),
  )

  return (
    <div style={styles.shell}>
      <Sidebar
        collapsed={isSidebarCollapsed}
        onToggleCollapse={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
        role="ADMIN"
        userName={userName}
        userEmail={userEmail}
        onLogout={onLogout}
        activeItem={activeView}
        onSelect={(item) => {
          setActiveView(item)
          if (item !== 'users') setSelectedUserId(null)
        }}
        pendingRequestsCount={pendingRequestsCount}
        conversations={conversations}
        activeConversationId={activeConversationId}
        onSelectConversation={handleSelectConversation}
        onNewChat={handleNewChat}
        onDeleteConversation={handleDeleteConversation}
        loadingConversations={loadingConversations}
      />

      <div style={styles.main}>
        {/* Toast */}
        {toast && (
          <div style={{ ...styles.toast, background: toast.type === 'error' ? '#fef2f2' : '#f0fdf4', borderColor: toast.type === 'error' ? '#fecaca' : '#bbf7d0', color: toast.type === 'error' ? '#991b1b' : '#166534' }}>
            {toast.message}
          </div>
        )}

        {activeView === 'chat' && (
          <RAG
            token={token}
            userName={userName}
            allowedUploadDepts={allowedUploadDepts}
            activeConversationId={activeConversationId}
            activeConversationTitle={conversations.find((c) => c.id === activeConversationId)?.title}
            onConversationCreatedOrUpdated={(convId) => {
              setActiveConversationId(convId)
              loadConversations()
            }}
            onToggleSidebar={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
            isSidebarCollapsed={isSidebarCollapsed}
          />
        )}

        {activeView === 'sources' && (
          <div style={styles.listContainer}>
            {/* Header bar */}
            <div style={styles.listHeader}>
              <div>
                <h1 style={styles.pageTitle}>Documents</h1>
                <p style={styles.pageSubtitle}>
                  {documents.length} document{documents.length !== 1 ? 's' : ''} · grounded knowledge base for RAG
                </p>
              </div>
            </div>

            {/* Filters */}
            <div style={styles.filterBar}>
              <div style={styles.searchWrap}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--color-ink-muted)" strokeWidth="2" strokeLinecap="round">
                  <circle cx="11" cy="11" r="8" />
                  <line x1="21" y1="21" x2="16.65" y2="16.65" />
                </svg>
                <input
                  placeholder="Search by file name, department, or uploader…"
                  value={docSearch}
                  onChange={(e) => setDocSearch(e.target.value)}
                  style={styles.searchInput}
                />
              </div>

              <select
                value={docDeptFilter}
                onChange={(e) => setDocDeptFilter(e.target.value)}
                style={styles.select}
              >
                <option value="ALL">All Departments</option>
                {departments.map((d) => (
                  <option key={d.code} value={d.code}>
                    {d.name} ({d.code})
                  </option>
                ))}
              </select>
            </div>

            {/* Documents Table */}
            <div style={styles.tableCard}>
              <table style={styles.table}>
                <thead>
                  <tr>
                    <th style={styles.th}>File Name</th>
                    <th style={styles.th}>Department</th>
                    <th style={styles.th}>Workspace</th>
                    <th style={styles.th}>Uploaded By</th>
                    <th style={styles.th}>Uploaded At</th>
                    <th style={styles.th}>Status</th>
                    <th style={{ ...styles.th, textAlign: 'right' }}>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {docsLoading ? (
                    <tr>
                      <td colSpan={7} style={{ ...styles.td, textAlign: 'center', color: 'var(--color-ink-muted)', padding: '32px' }}>
                        Loading documents…
                      </td>
                    </tr>
                  ) : filteredDocuments.length === 0 ? (
                    <tr>
                      <td colSpan={7} style={{ ...styles.td, textAlign: 'center', color: 'var(--color-ink-muted)', padding: '32px' }}>
                        No documents found.
                      </td>
                    </tr>
                  ) : (
                    filteredDocuments.map((doc) => (
                      <tr key={doc.id} style={styles.tr}>
                        <td style={styles.td}>
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
                        <td style={styles.td}>
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
                        <td style={styles.td}>
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
                        <td style={styles.td}>
                          <div style={{ fontSize: '13px', color: 'var(--color-ink)', fontWeight: 500 }}>
                            {doc.uploaded_by_name || (doc.uploaded_by_email ? doc.uploaded_by_email.split('@')[0] : 'Admin')}
                          </div>
                          {doc.uploaded_by_email && (
                            <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)', fontFamily: 'var(--font-mono)' }}>
                              {doc.uploaded_by_email}
                            </div>
                          )}
                        </td>
                        <td style={styles.td}>
                          <div style={{ fontSize: '12px', color: 'var(--color-ink)' }}>
                            {new Date(doc.created_at).toLocaleDateString()}
                          </div>
                          <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)' }}>
                            {new Date(doc.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                          </div>
                        </td>
                        <td style={styles.td}>
                          <span
                            style={{
                              display: 'inline-block',
                              padding: '3px 9px',
                              borderRadius: '12px',
                              background: '#ecfdf5',
                              color: '#047857',
                              border: '1px solid #a7f3d0',
                              fontSize: '12px',
                              fontWeight: 500,
                            }}
                          >
                            {doc.status || 'Ready'}
                          </span>
                        </td>
                        <td style={{ ...styles.td, textAlign: 'right' }}>
                          <div style={{ display: 'inline-flex', gap: '8px', alignItems: 'center' }}>
                            <button
                              onClick={() => handleOpenPreview(doc)}
                              style={{
                                ...styles.manageBtn,
                                background: 'var(--color-panel)',
                                color: 'var(--color-ink)',
                                display: 'inline-flex',
                                alignItems: 'center',
                                gap: '4px',
                                fontSize: '12px',
                                padding: '5px 10px',
                              }}
                            >
                              👁 Preview
                            </button>
                            <button
                              onClick={() => handleOpenManageAccess(doc)}
                              style={{
                                ...styles.manageBtn,
                                background: 'var(--color-teal-light)',
                                color: 'var(--color-teal-dim)',
                                border: '1px solid var(--color-teal)',
                                display: 'inline-flex',
                                alignItems: 'center',
                                gap: '4px',
                                fontSize: '12px',
                                padding: '5px 10px',
                                fontWeight: 500,
                              }}
                            >
                              🔒 Manage Access
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}


        {activeView === 'users' && !selectedUser && (
          <div style={styles.listContainer}>
            {/* Header bar */}
            <div style={styles.listHeader}>
              <div>
                <h1 style={styles.pageTitle}>User Management</h1>
                <p style={styles.pageSubtitle}>{users.length} users · manage roles, departments & permissions</p>
              </div>
              <button onClick={() => setShowCreateModal(true)} style={styles.createBtn}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" /></svg>
                Create User
              </button>
            </div>

            {/* Filters */}
            <div style={styles.filterBar}>
              <div style={styles.searchWrap}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--color-ink-muted)" strokeWidth="2" strokeLinecap="round"><circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" /></svg>
                <input
                  placeholder="Search by name or email…"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  style={styles.searchInput}
                />
              </div>
              <select value={roleFilter} onChange={(e) => setRoleFilter(e.target.value as any)} style={styles.select}>
                <option value="ALL">All Roles</option>
                <option value="ADMIN">Admin</option>
                <option value="USER">User</option>
              </select>
              <select value={accessFilter} onChange={(e) => setAccessFilter(e.target.value as any)} style={styles.select}>
                <option value="ALL">All Access</option>
                <option value="GLOBAL">Global Access</option>
                <option value="SCOPED">Department Access</option>
                <option value="NONE">No Access</option>
              </select>
            </div>

            {/* Table */}
            <div style={styles.tableWrap}>
              <table style={styles.table}>
                <thead>
                  <tr>
                    <th style={styles.th}>User</th>
                    <th style={styles.th}>Role</th>
                    <th style={styles.th}>Access</th>
                    <th style={styles.th}>Status</th>
                    <th style={{ ...styles.th, textAlign: 'right' }}>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredUsers.map((u) => {
                    const badge = accessBadgeColor(u)
                    return (
                      <tr key={u.id} style={styles.tr}>
                        <td style={styles.td}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                            <div style={styles.avatar}>{getDisplayName(u.email).charAt(0)}</div>
                            <div>
                              <div style={styles.userName}>{getDisplayName(u.email)}</div>
                              <div style={styles.userEmail}>{u.email}</div>
                            </div>
                          </div>
                        </td>
                        <td style={styles.td}>
                          <span style={{ ...styles.roleBadge, background: u.role === 'ADMIN' ? '#e8eaf6' : 'var(--color-ground)', color: u.role === 'ADMIN' ? '#283593' : 'var(--color-ink-muted)' }}>
                            {u.role}
                          </span>
                        </td>
                        <td style={styles.td}>
                          <span style={{ ...styles.accessBadge, background: badge.bg, color: badge.color }}>
                            {accessLabel(u)}
                          </span>
                        </td>
                        <td style={styles.td}>
                          <span style={{ ...styles.statusDot, background: u.is_active ? '#4caf50' : '#ef5350' }} />
                          <span style={{ fontSize: '13px', color: 'var(--color-ink-muted)' }}>{u.is_active ? 'Active' : 'Inactive'}</span>
                        </td>
                        <td style={{ ...styles.td, textAlign: 'right' }}>
                          <button onClick={() => selectUser(u.id)} style={styles.manageBtn}>
                            Manage Access
                          </button>
                        </td>
                      </tr>
                    )
                  })}
                  {filteredUsers.length === 0 && (
                    <tr>
                      <td colSpan={5} style={{ ...styles.td, textAlign: 'center', color: 'var(--color-ink-muted)', padding: '40px 16px' }}>
                        No users found matching your filters.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ====== USER DETAIL VIEW ====== */}
        {activeView === 'users' && selectedUser && (
          <div style={styles.detailContainer}>
            {/* Detail header */}
            <div style={styles.detailHeader}>
              <button onClick={() => setSelectedUserId(null)} style={styles.backBtn}>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="19" y1="12" x2="5" y2="12" /><polyline points="12 19 5 12 12 5" /></svg>
                Users
              </button>
              <div style={{ flex: 1 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <div style={{ ...styles.avatar, width: '48px', height: '48px', fontSize: '20px' }}>
                    {getDisplayName(selectedUser.email).charAt(0)}
                  </div>
                  <div>
                    <h1 style={{ margin: 0, fontFamily: 'var(--font-serif)', fontSize: '22px' }}>
                      {getDisplayName(selectedUser.email)}
                    </h1>
                    <div style={{ fontSize: '13px', color: 'var(--color-ink-muted)', display: 'flex', alignItems: 'center', gap: '8px', marginTop: '2px' }}>
                      {selectedUser.email}
                      <span style={{ ...styles.roleBadge, background: selectedUser.role === 'ADMIN' ? '#e8eaf6' : 'var(--color-ground)', color: selectedUser.role === 'ADMIN' ? '#283593' : 'var(--color-ink-muted)' }}>
                        {selectedUser.role}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Tabs */}
            <div style={styles.tabBar}>
              {(['profile', 'access', 'activity'] as const).map((tab) => (
                <button
                  key={tab}
                  onClick={() => setDetailTab(tab)}
                  style={{
                    ...styles.tab,
                    borderBottomColor: detailTab === tab ? 'var(--color-teal)' : 'transparent',
                    color: detailTab === tab ? 'var(--color-teal)' : 'var(--color-ink-muted)',
                    fontWeight: detailTab === tab ? 600 : 400,
                  }}
                >
                  {tab.charAt(0).toUpperCase() + tab.slice(1)}
                </button>
              ))}
            </div>

            {/* Tab content */}
            <div style={styles.tabContent}>
              {/* --- Profile Tab --- */}
              {detailTab === 'profile' && (
                <div style={styles.card}>
                  <h3 style={styles.cardTitle}>Profile Information</h3>
                  <div style={styles.fieldGrid}>
                    <div>
                      <div style={styles.fieldLabel}>Email</div>
                      <div style={styles.fieldValue}>{selectedUser.email}</div>
                    </div>
                    <div>
                      <div style={styles.fieldLabel}>Role</div>
                      <div style={styles.fieldValue}>{selectedUser.role}</div>
                    </div>
                    <div>
                      <div style={styles.fieldLabel}>Status</div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{ ...styles.statusDot, background: selectedUser.is_active ? '#4caf50' : '#ef5350' }} />
                        <span style={styles.fieldValue}>{selectedUser.is_active ? 'Active' : 'Inactive'}</span>
                        <button onClick={toggleStatus} disabled={saving} style={styles.smallActionBtn}>
                          {selectedUser.is_active ? 'Deactivate' : 'Activate'}
                        </button>
                      </div>
                    </div>
                    <div>
                      <div style={styles.fieldLabel}>Created</div>
                      <div style={styles.fieldValue}>{new Date(selectedUser.created_at).toLocaleDateString()}</div>
                    </div>
                    <div>
                      <div style={styles.fieldLabel}>User ID</div>
                      <div style={{ ...styles.fieldValue, fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--color-ink-muted)' }}>
                        {selectedUser.id}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* --- Access Tab --- */}
              {detailTab === 'access' && (
                <>
                  {selectedUser.role === 'ADMIN' ? (
                    <div style={styles.card}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '8px 0' }}>
                        <div style={{ width: '40px', height: '40px', borderRadius: '10px', background: '#e0f2f1', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#00695c" strokeWidth="2" strokeLinecap="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" /></svg>
                        </div>
                        <div>
                          <div style={{ fontWeight: 600, fontSize: '15px', color: 'var(--color-ink)' }}>Global Access</div>
                          <div style={{ fontSize: '13px', color: 'var(--color-ink-muted)' }}>
                            Admins have unrestricted access to all departments and permissions. No department assignment required.
                          </div>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <>
                      {/* Add Department */}
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px' }}>
                        <h3 style={{ ...styles.cardTitle, margin: 0 }}>
                          Department Access
                          <span style={{ fontWeight: 400, fontSize: '13px', color: 'var(--color-ink-muted)', marginLeft: '8px' }}>
                            {detailDepts.length} assigned
                          </span>
                        </h3>
                        {unassignedDepts.length > 0 && (
                          <div style={{ position: 'relative' }}>
                            <AddDepartmentDropdown
                              departments={unassignedDepts}
                              onSelect={assignDept}
                              disabled={saving}
                            />
                          </div>
                        )}
                      </div>

                      {detailDepts.length === 0 && (
                        <div style={styles.emptyState}>
                          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="var(--color-border)" strokeWidth="1.5" strokeLinecap="round"><circle cx="12" cy="12" r="10" /><line x1="15" y1="9" x2="9" y2="15" /><line x1="9" y1="9" x2="15" y2="15" /></svg>
                          <div style={{ marginTop: '12px', fontWeight: 500, color: 'var(--color-ink)' }}>No Access</div>
                          <div style={{ fontSize: '13px', color: 'var(--color-ink-muted)', marginTop: '4px' }}>
                            This user has no department assignments. Add a department to start granting permissions.
                          </div>
                        </div>
                      )}

                      {detailDepts.map((dept) => {
                        const myPerms = (detailPerms[dept.code] || []).map((p) => p.code)
                        const currentPreset = matchesPreset(myPerms)
                        return (
                          <div key={dept.code} style={styles.deptCard}>
                            <div style={styles.deptCardHeader}>
                              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                                <div style={styles.deptIcon}>
                                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--color-teal)" strokeWidth="2" strokeLinecap="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" /></svg>
                                </div>
                                <div>
                                  <div style={{ fontWeight: 600, fontSize: '14px', color: 'var(--color-ink)' }}>{dept.name || dept.code}</div>
                                  <div style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--color-ink-muted)', letterSpacing: '0.04em' }}>{dept.code}</div>
                                </div>
                              </div>
                              <button onClick={() => removeDept(dept.code)} disabled={saving} style={styles.removeDeptBtn}>
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>
                                Remove
                              </button>
                            </div>

                            {/* Presets */}
                            <div style={styles.presetRow}>
                              <span style={{ fontSize: '11px', color: 'var(--color-ink-muted)', fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Presets</span>
                              {Object.keys(PRESETS).map((p) => (
                                <button
                                  key={p}
                                  onClick={() => applyPreset(dept.code, p)}
                                  disabled={saving}
                                  style={{
                                    ...styles.presetBtn,
                                    background: currentPreset === p ? 'var(--color-teal)' : 'var(--color-ground)',
                                    color: currentPreset === p ? 'white' : 'var(--color-ink-muted)',
                                    borderColor: currentPreset === p ? 'var(--color-teal)' : 'var(--color-border)',
                                  }}
                                >
                                  {p}
                                </button>
                              ))}
                            </div>

                            {/* Permission checkboxes */}
                            <div style={styles.permGrid}>
                              {ALL_PERMS.map((p) => {
                                const has = myPerms.includes(p)
                                return (
                                  <label key={p} style={styles.permLabel}>
                                    <div
                                      style={{
                                        ...styles.checkbox,
                                        background: has ? 'var(--color-teal)' : 'transparent',
                                        borderColor: has ? 'var(--color-teal)' : 'var(--color-border)',
                                      }}
                                    >
                                      {has && (
                                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12" /></svg>
                                      )}
                                    </div>
                                    <span style={{ fontSize: '13px', fontFamily: 'var(--font-mono)', color: has ? 'var(--color-ink)' : 'var(--color-ink-muted)' }}>{p}</span>
                                    <input type="checkbox" checked={has} onChange={() => togglePerm(dept.code, p, has)} style={{ display: 'none' }} />
                                  </label>
                                )
                              })}
                            </div>
                          </div>
                        )
                      })}
                    </>
                  )}
                </>
              )}

              {/* --- Activity Tab --- */}
              {detailTab === 'activity' && (
                <div style={styles.emptyState}>
                  <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="var(--color-border)" strokeWidth="1.5" strokeLinecap="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12" /></svg>
                  <div style={{ marginTop: '12px', fontWeight: 500, color: 'var(--color-ink)' }}>Activity Log</div>
                  <div style={{ fontSize: '13px', color: 'var(--color-ink-muted)', marginTop: '4px' }}>
                    User activity tracking is coming soon. This will show login events, document actions, and permission changes.
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ====== DEPARTMENTS VIEW ====== */}
        {activeView === 'departments' && (
          <div style={styles.listContainer}>
            {/* Header bar */}
            <div style={styles.listHeader}>
              <div>
                <h1 style={styles.pageTitle}>Departments</h1>
                <p style={styles.pageSubtitle}>
                  {departments.length} departments · manage codes, names & activation status
                </p>
              </div>
              <button
                onClick={() => {
                  setCreateDeptCode('')
                  setCreateDeptName('')
                  setShowCreateDeptModal(true)
                }}
                style={styles.createBtn}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" /></svg>
                Create Department
              </button>
            </div>

            {/* Filter / Search Bar */}
            <div style={styles.filterBar}>
              <div style={styles.searchWrap}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--color-ink-muted)" strokeWidth="2" strokeLinecap="round"><circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" /></svg>
                <input
                  type="text"
                  placeholder="Search by code or name…"
                  value={deptSearch}
                  onChange={(e) => setDeptSearch(e.target.value)}
                  style={styles.searchInput}
                />
                {deptSearch && (
                  <button
                    onClick={() => setDeptSearch('')}
                    style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--color-ink-muted)', padding: '2px' }}
                  >
                    ✕
                  </button>
                )}
              </div>

              <select
                value={deptStatusFilter}
                onChange={(e) => setDeptStatusFilter(e.target.value as any)}
                style={styles.select}
              >
                <option value="ALL">All Statuses</option>
                <option value="ACTIVE">Active</option>
                <option value="INACTIVE">Inactive</option>
              </select>
            </div>

            {/* Table */}
            <div style={styles.tableWrap}>
              <table style={styles.table}>
                <thead>
                  <tr>
                    <th style={{ ...styles.th, width: '200px' }}>Code</th>
                    <th style={styles.th}>Name</th>
                    <th style={{ ...styles.th, width: '160px' }}>Status</th>
                    <th style={{ ...styles.th, width: '160px', textAlign: 'right' }}>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredDepartments.length === 0 ? (
                    <tr>
                      <td colSpan={4} style={{ textAlign: 'center', padding: '48px 16px', color: 'var(--color-ink-muted)' }}>
                        {departments.length === 0
                          ? 'No departments found. Click "+ Create Department" to create one.'
                          : 'No departments match your search or filter.'}
                      </td>
                    </tr>
                  ) : (
                    filteredDepartments.map((dept) => (
                      <tr key={dept.code} style={styles.tr}>
                        <td style={styles.td}>
                          <div style={{
                            display: 'inline-block',
                            padding: '4px 10px',
                            borderRadius: '6px',
                            background: 'var(--color-ground)',
                            border: '1px solid var(--color-border)',
                            fontFamily: 'var(--font-mono)',
                            fontSize: '12px',
                            fontWeight: 600,
                            color: 'var(--color-ink)',
                            letterSpacing: '0.04em',
                          }}>
                            {dept.code}
                          </div>
                        </td>
                        <td style={styles.td}>
                          <div style={{ fontWeight: 500, fontSize: '14px', color: 'var(--color-ink)' }}>
                            {dept.name}
                          </div>
                          {dept.created_at && (
                            <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)', marginTop: '2px' }}>
                              Created {new Date(dept.created_at).toLocaleDateString()}
                            </div>
                          )}
                        </td>
                        <td style={styles.td}>
                          <div style={{ display: 'flex', alignItems: 'center' }}>
                            <span style={{
                              ...styles.statusDot,
                              background: dept.is_active ? '#4caf50' : '#ef5350',
                            }} />
                            <span style={{
                              fontSize: '12px',
                              fontWeight: 500,
                              color: dept.is_active ? '#2e7d32' : '#c62828',
                            }}>
                              {dept.is_active ? 'Active' : 'Inactive'}
                            </span>
                          </div>
                        </td>
                        <td style={{ ...styles.td, textAlign: 'right' }}>
                          {dept.is_active ? (
                            <button
                              onClick={() => setDeactivateConfirmDept(dept)}
                              disabled={deptActionLoading}
                              style={{
                                ...styles.manageBtn,
                                color: '#c62828',
                                borderColor: '#ffcdd2',
                                background: '#ffebee',
                              }}
                            >
                              Deactivate
                            </button>
                          ) : (
                            <button
                              onClick={() => handleActivateDept(dept)}
                              disabled={deptActionLoading}
                              style={{
                                ...styles.manageBtn,
                                color: '#2e7d32',
                                borderColor: '#c8e6c9',
                                background: '#e8f5e9',
                              }}
                            >
                              Activate
                            </button>
                          )}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ====== ACCESS REQUESTS VIEW ====== */}
        {activeView === 'requests' && (
          <div style={styles.listContainer}>
            {/* Header bar */}
            <div style={styles.listHeader}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <h1 style={styles.pageTitle}>Access Requests</h1>
                  {pendingRequestsCount > 0 && (
                    <span style={{
                      background: '#fef3c7',
                      color: '#b45309',
                      border: '1px solid #fde68a',
                      fontSize: '12px',
                      fontWeight: 600,
                      padding: '2px 8px',
                      borderRadius: '12px',
                    }}>
                      {pendingRequestsCount} Pending
                    </span>
                  )}
                </div>
                <p style={styles.pageSubtitle}>
                  {accessRequests.length} total requests · review and approve department access requests
                </p>
              </div>
              <button
                onClick={loadAccessRequests}
                disabled={requestsLoading}
                style={{
                  ...styles.createBtn,
                  background: 'var(--color-surface)',
                  color: 'var(--color-ink)',
                  border: '1px solid var(--color-border)',
                }}
              >
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67" />
                </svg>
                {requestsLoading ? 'Refreshing…' : 'Refresh'}
              </button>
            </div>

            {/* Filter / Search Bar */}
            <div style={styles.filterBar}>
              <div style={styles.searchWrap}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--color-ink-muted)" strokeWidth="2" strokeLinecap="round"><circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" /></svg>
                <input
                  type="text"
                  placeholder="Search by requester, department or reason…"
                  value={requestSearch}
                  onChange={(e) => setRequestSearch(e.target.value)}
                  style={styles.searchInput}
                />
                {requestSearch && (
                  <button
                    onClick={() => setRequestSearch('')}
                    style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--color-ink-muted)', padding: '2px' }}
                  >
                    ✕
                  </button>
                )}
              </div>

              <select
                value={requestStatusFilter}
                onChange={(e) => setRequestStatusFilter(e.target.value as any)}
                style={styles.select}
              >
                <option value="ALL">All Statuses</option>
                <option value="PENDING">Pending Review</option>
                <option value="APPROVED">Approved</option>
                <option value="REJECTED">Rejected</option>
                <option value="CANCELLED">Cancelled</option>
              </select>
            </div>

            {/* Table */}
            <div style={styles.tableWrap}>
              <table style={styles.table}>
                <thead>
                  <tr>
                    <th style={styles.th}>Requester</th>
                    <th style={styles.th}>Department</th>
                    <th style={styles.th}>Requested Permissions</th>
                    <th style={{ ...styles.th, width: '140px' }}>Status</th>
                    <th style={{ ...styles.th, width: '160px' }}>Requested At</th>
                    <th style={{ ...styles.th, width: '160px', textAlign: 'right' }}>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredAccessRequests.length === 0 ? (
                    <tr>
                      <td colSpan={6} style={{ textAlign: 'center', padding: '48px 16px', color: 'var(--color-ink-muted)' }}>
                        {requestsLoading
                          ? 'Loading access requests…'
                          : accessRequests.length === 0
                          ? 'No access requests found in the system.'
                          : 'No access requests match your search or filter.'}
                      </td>
                    </tr>
                  ) : (
                    filteredAccessRequests.map((req) => {
                      const badge = accessRequestStatusBadge(req.status)
                      return (
                        <tr key={req.id} style={styles.tr}>
                          <td style={styles.td}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                              <div style={styles.avatar}>{getDisplayName(req.user_email).charAt(0)}</div>
                              <div>
                                <div style={styles.userName}>{getDisplayName(req.user_email)}</div>
                                <div style={styles.userEmail}>{req.user_email}</div>
                              </div>
                            </div>
                          </td>
                          <td style={styles.td}>
                            <div>
                              <div style={{ fontWeight: 600, fontSize: '14px', color: 'var(--color-ink)' }}>
                                {req.department_name}
                              </div>
                            </div>
                          </td>
                          <td style={styles.td}>
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
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
                                    fontWeight: 500,
                                  }}
                                >
                                  {p}
                                </span>
                              ))}
                            </div>
                          </td>
                          <td style={styles.td}>
                            <span style={{
                              display: 'inline-block',
                              padding: '3px 9px',
                              borderRadius: '12px',
                              background: badge.bg,
                              color: badge.color,
                              border: `1px solid ${badge.border}`,
                              fontSize: '12px',
                              fontWeight: 500,
                            }}>
                              {badge.label}
                            </span>
                          </td>
                          <td style={styles.td}>
                            <div style={{ fontSize: '12px', color: 'var(--color-ink)' }}>
                              {new Date(req.requested_at).toLocaleDateString()}
                            </div>
                            <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)' }}>
                              {new Date(req.requested_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                            </div>
                          </td>
                          <td style={{ ...styles.td, textAlign: 'right' }}>
                            {req.status === 'PENDING' ? (
                              <button
                                onClick={() => handleOpenReview(req)}
                                style={{
                                  ...styles.manageBtn,
                                  background: 'var(--color-teal)',
                                  color: 'white',
                                  borderColor: 'var(--color-teal)',
                                }}
                              >
                                Review Request
                              </button>
                            ) : (
                              <button
                                onClick={() => handleOpenReview(req)}
                                style={{
                                  ...styles.manageBtn,
                                  background: 'transparent',
                                  color: 'var(--color-ink-muted)',
                                }}
                              >
                                View Details
                              </button>
                            )}
                          </td>
                        </tr>
                      )
                    })
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {/* ====== CREATE USER MODAL ====== */}
      {showCreateModal && (
        <div style={styles.modalOverlay} onClick={() => setShowCreateModal(false)}>
          <div style={styles.modal} onClick={(e) => e.stopPropagation()}>
            <div style={styles.modalHeader}>
              <h2 style={{ margin: 0, fontFamily: 'var(--font-serif)', fontSize: '20px' }}>Create User</h2>
              <button onClick={() => setShowCreateModal(false)} style={styles.modalCloseBtn}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>
              </button>
            </div>

            <div style={styles.modalBody}>
              <div style={styles.formGroup}>
                <label style={styles.formLabel}>Email</label>
                <input
                  type="email"
                  placeholder="user@company.com"
                  value={createEmail}
                  onChange={(e) => setCreateEmail(e.target.value)}
                  style={styles.formInput}
                />
              </div>
              <div style={styles.formGroup}>
                <label style={styles.formLabel}>Password</label>
                <input
                  type="password"
                  placeholder="Minimum 8 characters"
                  value={createPassword}
                  onChange={(e) => setCreatePassword(e.target.value)}
                  style={styles.formInput}
                />
              </div>
              <div style={styles.formGroup}>
                <label style={styles.formLabel}>Role</label>
                <div style={{ display: 'flex', gap: '8px' }}>
                  <button
                    onClick={() => setCreateRole('USER')}
                    style={{
                      ...styles.roleToggle,
                      background: createRole === 'USER' ? 'var(--color-teal)' : 'var(--color-ground)',
                      color: createRole === 'USER' ? 'white' : 'var(--color-ink-muted)',
                      borderColor: createRole === 'USER' ? 'var(--color-teal)' : 'var(--color-border)',
                    }}
                  >
                    User
                  </button>
                  <button
                    onClick={() => setCreateRole('ADMIN')}
                    style={{
                      ...styles.roleToggle,
                      background: createRole === 'ADMIN' ? 'var(--color-teal)' : 'var(--color-ground)',
                      color: createRole === 'ADMIN' ? 'white' : 'var(--color-ink-muted)',
                      borderColor: createRole === 'ADMIN' ? 'var(--color-teal)' : 'var(--color-border)',
                    }}
                  >
                    Admin
                  </button>
                </div>
              </div>

              {/* Contextual info & Initial Access based on role */}
              {createRole === 'USER' && (
                <div style={{
                  padding: '16px',
                  borderRadius: '10px',
                  background: 'var(--color-ground)',
                  border: '1px solid var(--color-border)',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '12px',
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <label style={styles.formLabel}>Initial Department Access</label>
                    <span style={{ fontSize: '11px', color: 'var(--color-ink-muted)' }}>Optional</span>
                  </div>

                  <select
                    value={createInitialDept}
                    onChange={(e) => setCreateInitialDept(e.target.value)}
                    style={styles.formInput}
                  >
                    <option value="">-- No Initial Access (Assign Later) --</option>
                    {departments.filter(d => d.is_active).map((d) => (
                      <option key={d.code} value={d.code}>
                        {d.name}
                      </option>
                    ))}
                  </select>

                  {createInitialDept ? (
                    <div>
                      <div style={{ ...styles.formLabel, marginBottom: '6px' }}>Initial Permissions</div>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '8px' }}>
                        {ALL_PERMS.map((p) => {
                          const checked = createInitialPerms.includes(p)
                          return (
                            <label
                              key={p}
                              style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: '8px',
                                padding: '8px 10px',
                                borderRadius: '6px',
                                border: `1px solid ${checked ? 'var(--color-teal)' : 'var(--color-border)'}`,
                                background: checked ? 'rgba(0, 105, 92, 0.04)' : 'var(--color-surface)',
                                cursor: 'pointer',
                                fontSize: '12px',
                                fontFamily: 'var(--font-mono)',
                                color: 'var(--color-ink)',
                              }}
                            >
                              <input
                                type="checkbox"
                                checked={checked}
                                onChange={() => {
                                  if (checked) {
                                    setCreateInitialPerms(createInitialPerms.filter((x) => x !== p))
                                  } else {
                                    setCreateInitialPerms([...createInitialPerms, p])
                                  }
                                }}
                                style={{ accentColor: 'var(--color-teal)' }}
                              />
                              {p}
                            </label>
                          )
                        })}
                      </div>

                      {/* Quick Presets */}
                      <div style={{ display: 'flex', gap: '6px', marginTop: '8px' }}>
                        {Object.entries(PRESETS).map(([name, codes]) => (
                          <button
                            key={name}
                            type="button"
                            onClick={() => setCreateInitialPerms([...codes])}
                            style={{
                              padding: '3px 8px',
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

                      <div style={{
                        marginTop: '10px',
                        padding: '8px 12px',
                        borderRadius: '6px',
                        background: '#e0f2fe',
                        border: '1px solid #bae6fd',
                        fontSize: '12px',
                        color: '#0369a1',
                      }}>
                        ✨ User will be created with immediate <strong>{createInitialDept}</strong> access ({createInitialPerms.length > 0 ? createInitialPerms.join(', ') : 'No permissions selected'}).
                      </div>
                    </div>
                  ) : (
                    <div style={{
                      padding: '10px 14px',
                      borderRadius: '8px',
                      background: '#fff8e1',
                      border: '1px solid #fff0b3',
                      fontSize: '12px',
                      color: '#795548',
                      lineHeight: '1.4',
                    }}>
                      <strong style={{ color: '#f57f17' }}>⚠ No access by default</strong>
                      <div style={{ marginTop: '2px' }}>
                        User will be created without any department access. You can assign access later via <strong>Manage Access</strong> or user can submit an <strong>Access Request</strong>.
                      </div>
                    </div>
                  )}
                </div>
              )}

              {createRole === 'ADMIN' && (
                <div style={{
                  padding: '12px 16px',
                  borderRadius: '8px',
                  background: '#e0f2f1',
                  border: '1px solid #b2dfdb',
                  fontSize: '13px',
                  lineHeight: '1.5',
                }}>
                  <strong style={{ color: '#00695c' }}>🛡 Global Access</strong>
                  <div style={{ color: '#004d40', marginTop: '4px' }}>
                    Admin users have unrestricted access to all departments and features. No department assignment needed.
                  </div>
                </div>
              )}
            </div>

            <div style={styles.modalFooter}>
              <button onClick={() => setShowCreateModal(false)} style={styles.cancelBtn}>Cancel</button>
              <button
                onClick={createUser}
                disabled={
                  !createEmail ||
                  !createPassword ||
                  createPassword.length < 8 ||
                  (createRole === 'USER' && !!createInitialDept && createInitialPerms.length === 0) ||
                  createLoading
                }
                style={{
                  ...styles.submitBtn,
                  opacity:
                    !createEmail ||
                    !createPassword ||
                    createPassword.length < 8 ||
                    (createRole === 'USER' && !!createInitialDept && createInitialPerms.length === 0)
                      ? 0.5
                      : 1,
                }}
              >
                {createLoading ? 'Creating…' : 'Create User'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ====== CREATE DEPARTMENT MODAL ====== */}
      {showCreateDeptModal && (
        <div style={styles.modalOverlay} onClick={() => setShowCreateDeptModal(false)}>
          <div style={styles.modal} onClick={(e) => e.stopPropagation()}>
            <div style={styles.modalHeader}>
              <h2 style={{ margin: 0, fontFamily: 'var(--font-serif)', fontSize: '20px' }}>Create Department</h2>
              <button onClick={() => setShowCreateDeptModal(false)} style={styles.modalCloseBtn}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><line x1="18" y1="6" x2="6" y2="18" /></svg>
              </button>
            </div>

            <div style={styles.modalBody}>
              <div style={styles.formGroup}>
                <label style={styles.formLabel}>Department Code</label>
                <input
                  type="text"
                  placeholder="e.g. MARKETING, HR, FINANCE"
                  value={createDeptCode}
                  onChange={(e) => setCreateDeptCode(e.target.value.toUpperCase())}
                  style={styles.formInput}
                  autoFocus
                />
                <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)', marginTop: '4px' }}>
                  Uppercase identifier for authorization scopes.
                </div>
              </div>
              <div style={styles.formGroup}>
                <label style={styles.formLabel}>Department Name</label>
                <input
                  type="text"
                  placeholder="e.g. Marketing & Communications"
                  value={createDeptName}
                  onChange={(e) => setCreateDeptName(e.target.value)}
                  style={styles.formInput}
                />
              </div>

              <div style={{
                padding: '12px 16px',
                borderRadius: '8px',
                background: '#e0f2f1',
                border: '1px solid #b2dfdb',
                fontSize: '13px',
                lineHeight: '1.5',
              }}>
                <strong style={{ color: '#00695c' }}>ℹ Activation</strong>
                <div style={{ color: '#004d40', marginTop: '4px' }}>
                  The department will be created in <strong>Active</strong> status and immediately available for user assignment.
                </div>
              </div>
            </div>

            <div style={styles.modalFooter}>
              <button onClick={() => setShowCreateDeptModal(false)} style={styles.cancelBtn}>Cancel</button>
              <button
                onClick={handleCreateDept}
                disabled={!createDeptCode.trim() || !createDeptName.trim() || createDeptLoading}
                style={{
                  ...styles.submitBtn,
                  opacity: !createDeptCode.trim() || !createDeptName.trim() ? 0.5 : 1,
                }}
              >
                {createDeptLoading ? 'Creating…' : 'Create Department'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ====== DEACTIVATE DEPARTMENT CONFIRMATION MODAL ====== */}
      {deactivateConfirmDept && (
        <div style={styles.modalOverlay} onClick={() => setDeactivateConfirmDept(null)}>
          <div style={styles.modal} onClick={(e) => e.stopPropagation()}>
            <div style={styles.modalHeader}>
              <h2 style={{ margin: 0, fontFamily: 'var(--font-serif)', fontSize: '20px', color: '#c62828' }}>
                Deactivate Department
              </h2>
              <button onClick={() => setDeactivateConfirmDept(null)} style={styles.modalCloseBtn}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><line x1="18" y1="6" x2="6" y2="18" /></svg>
              </button>
            </div>

            <div style={styles.modalBody}>
              <p style={{ margin: '0 0 12px', fontSize: '14px', lineHeight: '1.5', color: 'var(--color-ink)' }}>
                Are you sure you want to deactivate department <strong>{deactivateConfirmDept.code}</strong> ({deactivateConfirmDept.name})?
              </p>

              <div style={{
                padding: '12px 16px',
                borderRadius: '8px',
                background: '#fff3e0',
                border: '1px solid #ffe0b2',
                fontSize: '13px',
                lineHeight: '1.5',
              }}>
                <strong style={{ color: '#e65100' }}>⚠ Safety Notice</strong>
                <div style={{ color: '#bf360c', marginTop: '4px' }}>
                  Deactivated departments will no longer be available for new user assignments or permission grants. Existing document references and membership records will remain preserved.
                </div>
              </div>
            </div>

            <div style={styles.modalFooter}>
              <button onClick={() => setDeactivateConfirmDept(null)} style={styles.cancelBtn}>Cancel</button>
              <button
                onClick={handleDeactivateDept}
                disabled={deptActionLoading}
                style={{
                  ...styles.submitBtn,
                  background: '#c62828',
                }}
              >
                {deptActionLoading ? 'Deactivating…' : 'Deactivate Department'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ====== ACCESS REQUEST REVIEW MODAL ====== */}
      {reviewModalRequest && (
        <div style={styles.modalOverlay} onClick={() => setReviewModalRequest(null)}>
          <div style={{ ...styles.modal, maxWidth: '560px' }} onClick={(e) => e.stopPropagation()}>
            <div style={styles.modalHeader}>
              <h2 style={{ margin: 0, fontFamily: 'var(--font-serif)', fontSize: '20px' }}>
                {reviewModalRequest.status === 'PENDING' ? 'Review Access Request' : 'Access Request Details'}
              </h2>
              <button onClick={() => setReviewModalRequest(null)} style={styles.modalCloseBtn}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><line x1="18" y1="6" x2="6" y2="18" /></svg>
              </button>
            </div>

            <div style={styles.modalBody}>
              {/* Requester & Department info cards */}
              <div style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '12px',
                padding: '14px',
                background: 'var(--color-ground)',
                borderRadius: '8px',
                border: '1px solid var(--color-border)',
                marginBottom: '16px',
              }}>
                <div>
                  <div style={styles.fieldLabel}>Requester</div>
                  <div style={{ fontWeight: 600, fontSize: '13px', color: 'var(--color-ink)' }}>
                    {reviewModalRequest.user_email}
                  </div>
                </div>
                <div>
                  <div style={styles.fieldLabel}>Department</div>
                  <div style={{ fontWeight: 600, fontSize: '13px', color: 'var(--color-ink)' }}>
                    {reviewModalRequest.department_name}
                  </div>
                </div>
                <div>
                  <div style={styles.fieldLabel}>Requested At</div>
                  <div style={{ fontSize: '12px', color: 'var(--color-ink)' }}>
                    {new Date(reviewModalRequest.requested_at).toLocaleString()}
                  </div>
                </div>
                <div>
                  <div style={styles.fieldLabel}>Status</div>
                  <div>
                    {(() => {
                      const b = accessRequestStatusBadge(reviewModalRequest.status)
                      return (
                        <span style={{
                          display: 'inline-block',
                          padding: '2px 8px',
                          borderRadius: '12px',
                          background: b.bg,
                          color: b.color,
                          border: `1px solid ${b.border}`,
                          fontSize: '11px',
                          fontWeight: 600,
                        }}>
                          {b.label}
                        </span>
                      )
                    })()}
                  </div>
                </div>
              </div>

              {/* Reason */}
              {reviewModalRequest.reason && (
                <div style={{ marginBottom: '16px' }}>
                  <div style={styles.fieldLabel}>Requester Reason / Justification</div>
                  <div style={{
                    padding: '10px 14px',
                    borderRadius: '8px',
                    background: '#f8fafc',
                    border: '1px solid #e2e8f0',
                    fontSize: '13px',
                    color: '#334155',
                    fontStyle: 'italic',
                    lineHeight: '1.5',
                  }}>
                    "{reviewModalRequest.reason}"
                  </div>
                </div>
              )}

              {/* Requested permissions */}
              <div style={{ marginBottom: '16px' }}>
                <div style={styles.fieldLabel}>Originally Requested Permissions</div>
                <div style={{ display: 'flex', gap: '6px', marginTop: '4px' }}>
                  {reviewModalRequest.requested_permissions.map((p) => (
                    <span
                      key={p}
                      style={{
                        fontSize: '12px',
                        fontFamily: 'var(--font-mono)',
                        padding: '3px 8px',
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
              </div>

              {/* Review Section (if PENDING) */}
              {reviewModalRequest.status === 'PENDING' ? (
                <>
                  <div style={styles.formGroup}>
                    <label style={styles.formLabel}>
                      Final Approved Permissions
                      <span style={{ fontSize: '11px', fontWeight: 400, color: 'var(--color-ink-muted)', marginLeft: '8px' }}>
                        (You can modify the granted permissions)
                      </span>
                    </label>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '8px', marginTop: '6px' }}>
                      {ALL_PERMS.map((p) => {
                        const isApproved = approvedPerms.includes(p)
                        return (
                          <label
                            key={p}
                            style={{
                              display: 'flex',
                              alignItems: 'center',
                              gap: '10px',
                              padding: '10px 12px',
                              borderRadius: '8px',
                              border: `1px solid ${isApproved ? 'var(--color-teal)' : 'var(--color-border)'}`,
                              background: isApproved ? 'rgba(0, 105, 92, 0.04)' : 'var(--color-surface)',
                              cursor: 'pointer',
                            }}
                          >
                            <input
                              type="checkbox"
                              checked={isApproved}
                              onChange={() => handleToggleApprovedPerm(p)}
                              style={{ accentColor: 'var(--color-teal)' }}
                            />
                            <div>
                              <div style={{ fontWeight: 600, fontSize: '13px', fontFamily: 'var(--font-mono)', color: 'var(--color-ink)' }}>
                                {p}
                              </div>
                            </div>
                          </label>
                        )
                      })}
                    </div>
                  </div>

                  <div style={styles.formGroup}>
                    <label style={styles.formLabel}>
                      Review Note (Optional)
                    </label>
                    <textarea
                      rows={2}
                      placeholder="Add an optional note or explanation for the approval/rejection decision…"
                      value={reviewNote}
                      onChange={(e) => setReviewNote(e.target.value)}
                      style={{
                        ...styles.formInput,
                        fontFamily: 'var(--font-sans)',
                        resize: 'vertical',
                      }}
                    />
                  </div>
                </>
              ) : (
                /* Historical Review details */
                <div style={{
                  padding: '12px 14px',
                  borderRadius: '8px',
                  background: '#f8fafc',
                  border: '1px solid #e2e8f0',
                  marginTop: '12px',
                }}>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                    {reviewModalRequest.reviewer_email && (
                      <div>
                        <div style={styles.fieldLabel}>Reviewed By</div>
                        <div style={{ fontSize: '13px', color: 'var(--color-ink)', fontWeight: 500 }}>
                          {reviewModalRequest.reviewer_email}
                        </div>
                      </div>
                    )}
                    {reviewModalRequest.reviewed_at && (
                      <div>
                        <div style={styles.fieldLabel}>Reviewed At</div>
                        <div style={{ fontSize: '13px', color: 'var(--color-ink)' }}>
                          {new Date(reviewModalRequest.reviewed_at).toLocaleString()}
                        </div>
                      </div>
                    )}
                  </div>
                  {reviewModalRequest.review_note && (
                    <div style={{ marginTop: '10px' }}>
                      <div style={styles.fieldLabel}>Review Note</div>
                      <div style={{ fontSize: '13px', color: '#334155', marginTop: '2px' }}>
                        {reviewModalRequest.review_note}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            <div style={styles.modalFooter}>
              {reviewModalRequest.status === 'PENDING' ? (
                <>
                  <button
                    onClick={() => setReviewModalRequest(null)}
                    disabled={reviewSubmitting}
                    style={styles.cancelBtn}
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleRejectRequest}
                    disabled={reviewSubmitting}
                    style={{
                      ...styles.submitBtn,
                      background: '#c62828',
                    }}
                  >
                    {reviewSubmitting ? 'Rejecting…' : 'Reject Request'}
                  </button>
                  <button
                    onClick={handleApproveRequest}
                    disabled={approvedPerms.length === 0 || reviewSubmitting}
                    style={{
                      ...styles.submitBtn,
                      opacity: approvedPerms.length === 0 ? 0.5 : 1,
                    }}
                  >
                    {reviewSubmitting ? 'Approving…' : 'Approve Request'}
                  </button>
                </>
              ) : (
                <button
                  onClick={() => setReviewModalRequest(null)}
                  style={styles.submitBtn}
                >
                  Close
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ====== PREVIEW DOCUMENT MODAL ====== */}
      {previewDoc && (
        <div style={styles.modalOverlay} onClick={() => setPreviewDoc(null)}>
          <div
            style={{
              ...styles.modal,
              maxWidth: '800px',
              width: '90%',
              maxHeight: '85vh',
              display: 'flex',
              flexDirection: 'column',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={styles.modalHeader}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '20px' }}>📄</span>
                <div>
                  <h2 style={{ margin: 0, fontFamily: 'var(--font-serif)', fontSize: '18px' }}>
                    {previewDoc.file_name}
                  </h2>
                  <div style={{ fontSize: '12px', color: 'var(--color-ink-muted)', marginTop: '2px' }}>
                    Department: <strong>{previewDoc.department_name || previewDoc.department_code || 'Unassigned'}</strong> · Uploaded by:{' '}
                    <strong>{previewDoc.uploaded_by_name || previewDoc.uploaded_by_email || 'Admin'}</strong>
                  </div>
                </div>
              </div>
              <button onClick={() => setPreviewDoc(null)} style={styles.modalCloseBtn}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>

            <div style={{ ...styles.modalBody, flex: 1, overflowY: 'auto', minHeight: '280px' }}>
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

            <div style={{ ...styles.modalFooter, justifyContent: 'space-between', padding: '12px 24px', borderTop: '1px solid var(--color-border)' }}>
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

      {/* ====== MANAGE ACCESS MODAL ====== */}
      {manageAccessDoc && (
        <div style={styles.modalOverlay} onClick={() => setManageAccessDoc(null)}>
          <div
            style={{
              ...styles.modal,
              maxWidth: '850px',
              width: '92%',
              maxHeight: '85vh',
              display: 'flex',
              flexDirection: 'column',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={styles.modalHeader}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <span style={{ fontSize: '22px' }}>🔒</span>
                <div>
                  <h2 style={{ margin: 0, fontFamily: 'var(--font-serif)', fontSize: '18px' }}>
                    Document Access: {manageAccessDoc.file_name}
                  </h2>
                  <div style={{ fontSize: '12px', color: 'var(--color-ink-muted)', marginTop: '2px' }}>
                    Department: <strong>{manageAccessDoc.department_name || manageAccessDoc.department_code}</strong> · Manage specific user permissions
                  </div>
                </div>
              </div>
              <button onClick={() => setManageAccessDoc(null)} style={styles.modalCloseBtn}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>

            <div style={{ ...styles.modalBody, flex: 1, overflowY: 'auto', padding: '20px' }}>
              {docAccessLoading ? (
                <div style={{ textAlign: 'center', padding: '48px', color: 'var(--color-ink-muted)' }}>
                  Loading department members & access rules…
                </div>
              ) : !docAccessData || docAccessData.users.length === 0 ? (
                <div style={{ textAlign: 'center', padding: '32px', color: 'var(--color-ink-muted)' }}>
                  No active users found in department {manageAccessDoc.department_code}.
                </div>
              ) : (
                <div>
                  <p style={{ fontSize: '13px', color: 'var(--color-ink-muted)', marginBottom: '16px' }}>
                    Users in <strong>{docAccessData.department_name}</strong> department can be granted specific permissions on this document. Department-level permissions already granted are shown with a checkmark.
                  </p>

                  <div style={{ border: '1px solid var(--color-border)', borderRadius: '8px', overflow: 'hidden' }}>
                    <table style={{ ...styles.table, margin: 0 }}>
                      <thead>
                        <tr style={{ background: 'var(--color-ground)' }}>
                          <th style={styles.th}>User</th>
                          <th style={styles.th}>Dept Perms</th>
                          <th style={styles.th}>Document Permissions</th>
                          <th style={styles.th}>Effective</th>
                          <th style={{ ...styles.th, textAlign: 'right' }}>Action</th>
                        </tr>
                      </thead>
                      <tbody>
                        {docAccessData.users.map((u) => {
                          const userPendingPerms = pendingDocPerms[u.user_id] || []
                          const isSaving = docAccessSavingId === u.user_id

                          return (
                            <tr key={u.user_id} style={styles.tr}>
                              <td style={styles.td}>
                                <div style={{ fontWeight: 600, color: 'var(--color-ink)', fontSize: '13px' }}>
                                  {u.full_name || u.email.split('@')[0]}
                                </div>
                                <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)', fontFamily: 'var(--font-mono)' }}>
                                  {u.email}
                                </div>
                              </td>

                              <td style={styles.td}>
                                {u.department_permissions.length === 0 ? (
                                  <span style={{ fontSize: '11px', color: 'var(--color-ink-muted)' }}>None</span>
                                ) : (
                                  <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
                                    {u.department_permissions.map((p) => (
                                      <span
                                        key={p}
                                        style={{
                                          fontSize: '10px',
                                          padding: '2px 6px',
                                          borderRadius: '4px',
                                          background: 'var(--color-teal-light)',
                                          color: 'var(--color-teal)',
                                          fontWeight: 600,
                                        }}
                                      >
                                        {p}
                                      </span>
                                    ))}
                                  </div>
                                )}
                              </td>

                              <td style={styles.td}>
                                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
                                  {ALL_PERMS.map((p) => {
                                    const isChecked = userPendingPerms.includes(p)
                                    const isDeptGranted = u.department_permissions.includes(p)

                                    return (
                                      <label
                                        key={p}
                                        style={{
                                          display: 'inline-flex',
                                          alignItems: 'center',
                                          gap: '4px',
                                          fontSize: '12px',
                                          cursor: 'pointer',
                                          padding: '3px 8px',
                                          borderRadius: '6px',
                                          background: isChecked ? 'var(--color-teal-light)' : 'var(--color-ground)',
                                          border: isChecked ? '1px solid var(--color-teal)' : '1px solid var(--color-border)',
                                          color: isChecked ? 'var(--color-teal-dim)' : 'var(--color-ink)',
                                          fontWeight: isChecked ? 600 : 400,
                                        }}
                                      >
                                        <input
                                          type="checkbox"
                                          checked={isChecked}
                                          onChange={() => handleToggleDocPerm(u.user_id, p)}
                                          style={{ cursor: 'pointer', accentColor: 'var(--color-teal)' }}
                                        />
                                        {p}
                                        {isDeptGranted && (
                                          <span style={{ fontSize: '10px', color: '#059669', fontWeight: 'bold' }} title="Already granted via department">
                                            ✓
                                          </span>
                                        )}
                                      </label>
                                    )
                                  })}
                                </div>
                              </td>

                              <td style={styles.td}>
                                <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
                                  {Array.from(new Set([...u.department_permissions, ...userPendingPerms])).map((p) => (
                                    <span
                                      key={p}
                                      style={{
                                        fontSize: '10px',
                                        padding: '2px 6px',
                                        borderRadius: '4px',
                                        background: '#ecfdf5',
                                        color: '#047857',
                                        fontWeight: 600,
                                        border: '1px solid #a7f3d0',
                                      }}
                                    >
                                      {p}
                                    </span>
                                  ))}
                                </div>
                              </td>

                              <td style={{ ...styles.td, textAlign: 'right' }}>
                                <button
                                  onClick={() => handleSaveUserDocPerms(u.user_id)}
                                  disabled={isSaving}
                                  style={{
                                    ...styles.manageBtn,
                                    background: 'var(--color-teal)',
                                    color: '#ffffff',
                                    border: 'none',
                                    fontSize: '11px',
                                    padding: '5px 12px',
                                    opacity: isSaving ? 0.7 : 1,
                                    cursor: isSaving ? 'not-allowed' : 'pointer',
                                  }}
                                >
                                  {isSaving ? 'Saving…' : 'Save'}
                                </button>
                              </td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>

            <div style={{ ...styles.modalFooter, justifyContent: 'space-between', padding: '12px 24px', borderTop: '1px solid var(--color-border)' }}>
              <span style={{ fontSize: '11px', color: 'var(--color-ink-muted)' }}>
                Document ID: <code style={{ fontFamily: 'var(--font-mono)' }}>{manageAccessDoc.id}</code>
              </span>
              <button onClick={() => setManageAccessDoc(null)} style={styles.cancelBtn}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/*  Add Department Dropdown                                            */
/* ------------------------------------------------------------------ */

function AddDepartmentDropdown({
  departments,
  onSelect,
  disabled,
}: {
  departments: DepartmentResponse[]
  onSelect: (code: string) => void
  disabled: boolean
}) {
  const [open, setOpen] = useState(false)
  return (
    <div style={{ position: 'relative' }}>
      <button onClick={() => setOpen(!open)} disabled={disabled} style={styles.addDeptBtn}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" /></svg>
        Add Department
      </button>
      {open && (
        <div style={styles.dropdown}>
          {departments.map((d) => (
            <button
              key={d.code}
              onClick={() => {
                onSelect(d.code)
                setOpen(false)
              }}
              style={styles.dropdownItem}
              onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--color-ground)')}
              onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
            >
              <span style={{ fontWeight: 500 }}>{d.name || d.code}</span>
              <span style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--color-ink-muted)' }}>{d.code}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/*  Styles                                                             */
/* ------------------------------------------------------------------ */

const styles: Record<string, React.CSSProperties> = {
  shell: {
    display: 'flex',
    height: '100vh',
    maxHeight: '100vh',
    overflow: 'hidden',
    background: 'var(--color-ground)',
    fontFamily: 'var(--font-sans)',
    color: 'var(--color-ink)',
  },
  main: {
    flex: 1,
    minWidth: 0,
    height: '100vh',
    maxHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
    position: 'relative',
    overflow: 'hidden',
  },
  hamburger: { position: 'absolute', top: '10px', left: '16px', zIndex: 10 },
  hamburgerBtn: {
    width: '36px',
    height: '36px',
    borderRadius: '8px',
    background: 'var(--color-surface)',
    border: '1px solid var(--color-border)',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  toast: {
    position: 'fixed',
    top: '16px',
    right: '16px',
    padding: '12px 20px',
    borderRadius: '10px',
    border: '1px solid',
    fontSize: '13px',
    fontWeight: 500,
    zIndex: 2000,
    boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
    animation: 'fadeIn 0.2s ease',
  },

  /* LIST VIEW */
  listContainer: {
    flex: 1,
    minHeight: 0,
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    padding: '24px 32px',
    overflowY: 'auto',
    overflowX: 'hidden',
  },
  listHeader: {
    display: 'flex',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    marginBottom: '20px',
  },
  pageTitle: {
    margin: '0 0 4px',
    fontFamily: 'var(--font-serif)',
    fontSize: '24px',
  },
  pageSubtitle: {
    margin: 0,
    fontSize: '13px',
    color: 'var(--color-ink-muted)',
  },
  createBtn: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    padding: '10px 20px',
    background: 'var(--color-teal)',
    color: 'white',
    border: 'none',
    borderRadius: '10px',
    fontSize: '14px',
    fontWeight: 600,
    cursor: 'pointer',
    transition: 'background 0.15s',
    whiteSpace: 'nowrap',
  },
  filterBar: {
    display: 'flex',
    gap: '10px',
    marginBottom: '16px',
    flexWrap: 'wrap',
  },
  searchWrap: {
    flex: 1,
    minWidth: '200px',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '0 14px',
    height: '40px',
    background: 'var(--color-surface)',
    border: '1px solid var(--color-border)',
    borderRadius: '10px',
  },
  searchInput: {
    flex: 1,
    background: 'transparent',
    border: 'none',
    fontSize: '13px',
    color: 'var(--color-ink)',
    fontFamily: 'var(--font-sans)',
  },
  select: {
    height: '40px',
    padding: '0 12px',
    border: '1px solid var(--color-border)',
    borderRadius: '10px',
    background: 'var(--color-surface)',
    fontSize: '13px',
    color: 'var(--color-ink)',
    fontFamily: 'var(--font-sans)',
    cursor: 'pointer',
    appearance: 'auto' as any,
  },

  /* TABLE */
  tableWrap: {
    width: '100%',
    overflowX: 'auto',
    background: 'var(--color-surface)',
    borderRadius: '12px',
    border: '1px solid var(--color-border)',
  },
  tableCard: {
    width: '100%',
    overflowX: 'auto',
    background: 'var(--color-surface)',
    borderRadius: '12px',
    border: '1px solid var(--color-border)',
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse',
    fontSize: '14px',
  },
  th: {
    textAlign: 'left' as const,
    padding: '12px 16px',
    fontSize: '11px',
    fontFamily: 'var(--font-mono)',
    fontWeight: 500,
    color: 'var(--color-ink-muted)',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.06em',
    borderBottom: '1px solid var(--color-border)',
    background: 'var(--color-ground)',
    position: 'sticky' as const,
    top: 0,
    zIndex: 1,
  },
  tr: {
    borderBottom: '1px solid var(--color-border)',
    transition: 'background 0.1s',
  },
  td: {
    padding: '14px 16px',
    verticalAlign: 'middle' as const,
  },
  avatar: {
    width: '36px',
    height: '36px',
    borderRadius: '50%',
    background: 'var(--color-teal)',
    color: 'white',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '15px',
    fontWeight: 600,
    flexShrink: 0,
  },
  userName: { fontSize: '14px', fontWeight: 500, color: 'var(--color-ink)' },
  userEmail: { fontSize: '12px', color: 'var(--color-ink-muted)' },
  roleBadge: {
    fontSize: '10px',
    fontFamily: 'var(--font-mono)',
    padding: '3px 8px',
    borderRadius: '6px',
    letterSpacing: '0.04em',
    fontWeight: 600,
  },
  accessBadge: {
    fontSize: '11px',
    fontFamily: 'var(--font-mono)',
    padding: '4px 10px',
    borderRadius: '6px',
    letterSpacing: '0.02em',
    display: 'inline-block',
    maxWidth: '320px',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  statusDot: {
    width: '8px',
    height: '8px',
    borderRadius: '50%',
    display: 'inline-block',
    marginRight: '6px',
  },
  manageBtn: {
    padding: '6px 14px',
    border: '1px solid var(--color-border)',
    borderRadius: '8px',
    background: 'var(--color-surface)',
    fontSize: '12px',
    fontWeight: 500,
    color: 'var(--color-teal)',
    cursor: 'pointer',
    transition: 'border-color 0.15s, background 0.15s',
    whiteSpace: 'nowrap',
  },

  /* DETAIL VIEW */
  detailContainer: {
    flex: 1,
    minHeight: 0,
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    padding: '24px 32px 24px 64px',
    overflowY: 'auto',
    overflowX: 'hidden',
  },
  detailHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '16px',
    marginBottom: '20px',
  },
  backBtn: {
    display: 'flex',
    alignItems: 'center',
    gap: '4px',
    background: 'transparent',
    border: 'none',
    cursor: 'pointer',
    color: 'var(--color-ink-muted)',
    fontSize: '13px',
    padding: '6px 10px',
    borderRadius: '8px',
    transition: 'color 0.15s',
    flexShrink: 0,
  },
  tabBar: {
    display: 'flex',
    gap: '0',
    borderBottom: '1px solid var(--color-border)',
    marginBottom: '24px',
  },
  tab: {
    padding: '10px 20px',
    background: 'transparent',
    border: 'none',
    borderBottom: '2px solid transparent',
    fontSize: '14px',
    cursor: 'pointer',
    transition: 'color 0.15s',
  },
  tabContent: {
    flex: 1,
    overflowY: 'auto',
    paddingBottom: '40px',
  },
  card: {
    background: 'var(--color-surface)',
    border: '1px solid var(--color-border)',
    borderRadius: '12px',
    padding: '24px',
    marginBottom: '16px',
  },
  cardTitle: {
    fontSize: '15px',
    fontWeight: 600,
    marginBottom: '16px',
    color: 'var(--color-ink)',
  },
  fieldGrid: {
    display: 'grid',
    gap: '16px',
  },
  fieldLabel: {
    fontSize: '11px',
    fontFamily: 'var(--font-mono)',
    color: 'var(--color-ink-muted)',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.06em',
    marginBottom: '4px',
  },
  fieldValue: {
    fontSize: '14px',
    color: 'var(--color-ink)',
  },
  smallActionBtn: {
    padding: '4px 12px',
    borderRadius: '6px',
    border: '1px solid var(--color-border)',
    background: 'var(--color-ground)',
    fontSize: '12px',
    cursor: 'pointer',
    color: 'var(--color-ink-muted)',
  },
  emptyState: {
    textAlign: 'center' as const,
    padding: '48px 24px',
    background: 'var(--color-surface)',
    border: '1px solid var(--color-border)',
    borderRadius: '12px',
  },

  /* DEPT CARD */
  deptCard: {
    background: 'var(--color-surface)',
    border: '1px solid var(--color-border)',
    borderRadius: '12px',
    padding: '20px',
    marginBottom: '12px',
  },
  deptCardHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '16px',
  },
  deptIcon: {
    width: '32px',
    height: '32px',
    borderRadius: '8px',
    background: 'var(--color-teal-light)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  removeDeptBtn: {
    display: 'flex',
    alignItems: 'center',
    gap: '4px',
    padding: '4px 10px',
    background: 'transparent',
    border: 'none',
    color: '#d93025',
    fontSize: '12px',
    cursor: 'pointer',
    borderRadius: '6px',
    transition: 'background 0.15s',
  },
  presetRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    marginBottom: '14px',
    paddingBottom: '14px',
    borderBottom: '1px solid var(--color-border)',
  },
  presetBtn: {
    padding: '4px 12px',
    borderRadius: '6px',
    border: '1px solid',
    fontSize: '11px',
    fontWeight: 500,
    cursor: 'pointer',
    transition: 'all 0.15s',
  },
  permGrid: {
    display: 'flex',
    gap: '24px',
    flexWrap: 'wrap',
  },
  permLabel: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    cursor: 'pointer',
  },
  checkbox: {
    width: '18px',
    height: '18px',
    borderRadius: '4px',
    border: '1.5px solid',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'all 0.15s',
  },
  addDeptBtn: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    padding: '8px 16px',
    background: 'var(--color-surface)',
    border: '1px solid var(--color-teal)',
    borderRadius: '8px',
    color: 'var(--color-teal)',
    fontSize: '13px',
    fontWeight: 500,
    cursor: 'pointer',
    transition: 'background 0.15s',
  },
  dropdown: {
    position: 'absolute' as const,
    top: '100%',
    right: 0,
    marginTop: '4px',
    background: 'var(--color-surface)',
    border: '1px solid var(--color-border)',
    borderRadius: '10px',
    boxShadow: '0 8px 24px rgba(0,0,0,0.1)',
    minWidth: '200px',
    zIndex: 100,
    padding: '4px',
    maxHeight: '200px',
    overflowY: 'auto',
  },
  dropdownItem: {
    width: '100%',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '10px 14px',
    background: 'transparent',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    fontSize: '13px',
    color: 'var(--color-ink)',
    textAlign: 'left' as const,
    transition: 'background 0.1s',
  },

  /* MODAL */
  modalOverlay: {
    position: 'fixed',
    inset: 0,
    background: 'rgba(21, 29, 50, 0.5)',
    backdropFilter: 'blur(4px)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 3000,
  },
  modal: {
    background: 'var(--color-surface)',
    borderRadius: '16px',
    width: '100%',
    maxWidth: '440px',
    boxShadow: '0 20px 60px rgba(0,0,0,0.15)',
    overflow: 'hidden',
  },
  modalHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '20px 24px',
    borderBottom: '1px solid var(--color-border)',
  },
  modalCloseBtn: {
    background: 'transparent',
    border: 'none',
    cursor: 'pointer',
    color: 'var(--color-ink-muted)',
    padding: '4px',
    borderRadius: '6px',
  },
  modalBody: {
    padding: '24px',
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
  },
  modalFooter: {
    display: 'flex',
    justifyContent: 'flex-end',
    gap: '8px',
    padding: '16px 24px',
    borderTop: '1px solid var(--color-border)',
  },
  formGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
  },
  formLabel: {
    fontSize: '12px',
    fontFamily: 'var(--font-mono)',
    fontWeight: 500,
    color: 'var(--color-ink-muted)',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.06em',
  },
  formInput: {
    padding: '10px 14px',
    borderRadius: '10px',
    border: '1px solid var(--color-border)',
    background: 'var(--color-ground)',
    fontSize: '14px',
    color: 'var(--color-ink)',
    fontFamily: 'var(--font-sans)',
  },
  roleToggle: {
    padding: '8px 20px',
    borderRadius: '8px',
    border: '1px solid',
    fontSize: '13px',
    fontWeight: 500,
    cursor: 'pointer',
    transition: 'all 0.15s',
  },
  cancelBtn: {
    padding: '8px 20px',
    borderRadius: '8px',
    border: '1px solid var(--color-border)',
    background: 'var(--color-ground)',
    fontSize: '13px',
    cursor: 'pointer',
    color: 'var(--color-ink-muted)',
  },
  submitBtn: {
    padding: '8px 24px',
    borderRadius: '8px',
    border: 'none',
    background: 'var(--color-teal)',
    color: 'white',
    fontSize: '13px',
    fontWeight: 600,
    cursor: 'pointer',
    transition: 'opacity 0.15s',
  },
}
