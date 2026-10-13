import React, { useState, useEffect, useRef, useMemo } from 'react'
import {
  CompanyNote,
  CompanyNoteEvidenceItem,
  DocumentChunkInfo,
  WebsiteSourceInput,
  fetchCompanyNotes,
  createCompanyNote,
  updateCompanyNote,
  attachNoteEvidence,
  removeNoteEvidence,
  shareNoteToWorkspace,
  submitCompanyNote,
  reviewCompanyNote,
  batchPublishCompanyNotes,
  archiveCompanyNote,
  supersedeCompanyNote,
  fetchDocumentChunks,
  BatchPublishResponse,
} from '../api/companyNotes'
import { listDocuments, DocumentListItem, getDocumentDisplayName } from '../api/documents'

interface Props {
  token: string
  userRole: string
  userName: string
  onToast: (msg: string, type?: 'success' | 'error') => void
  workspaceId?: string | null
}

export default function CompanyNotesView({ token, userRole, userName, onToast, workspaceId }: Props) {
  const isAdmin = userRole === 'ADMIN'

  const [notes, setNotes] = useState<CompanyNote[]>([])
  const [loading, setLoading] = useState<boolean>(true)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [isPermissionDenied, setIsPermissionDenied] = useState<boolean>(false)

  // Search input with debouncing to prevent stale request races
  const [searchInput, setSearchInput] = useState<string>('')
  const [search, setSearch] = useState<string>('')
  const [showMoreFilters, setShowMoreFilters] = useState<boolean>(false)

  const [filterStatus, setFilterStatus] = useState<string>('ALL')
  const [filterVisibility, setFilterVisibility] = useState<string>('ALL')
  const [filterObsType, setFilterObsType] = useState<string>('ALL')
  const [filterCompany, setFilterCompany] = useState<string>('ALL')
  const [filterWorkspaceOnly, setFilterWorkspaceOnly] = useState<boolean>(false)

  // Selected note for review / details
  const [selectedNote, setSelectedNote] = useState<CompanyNote | null>(null)

  // Modals
  const [showCreateModal, setShowCreateModal] = useState<boolean>(false)
  const [showAttachModal, setShowAttachModal] = useState<boolean>(false)
  const [showEditModal, setShowEditModal] = useState<boolean>(false)
  const [batchModalResult, setBatchModalResult] = useState<BatchPublishResponse | null>(null)

  // Create Form State
  const [createCompany, setCreateCompany] = useState<string>('')
  const [createStatement, setCreateStatement] = useState<string>('')
  const [createObsType, setCreateObsType] = useState<'SUPPORTED' | 'INFERRED' | 'UNVERIFIED'>('SUPPORTED')
  const [createVisibility, setCreateVisibility] = useState<'PRIVATE' | 'WORKSPACE'>('PRIVATE')

  // Create Form Staged Evidence
  const [createStagedDocs, setCreateStagedDocs] = useState<Array<{ id: string; fileName: string; excerpt: string; page?: number | null }>>([])
  const [createStagedWebsites, setCreateStagedWebsites] = useState<WebsiteSourceInput[]>([])
  const [createShowAddSource, setCreateShowAddSource] = useState<boolean>(false)

  // Edit Form State
  const [editCompany, setEditCompany] = useState<string>('')
  const [editStatement, setEditStatement] = useState<string>('')
  const [editObsType, setEditObsType] = useState<'SUPPORTED' | 'INFERRED' | 'UNVERIFIED'>('SUPPORTED')
  const [editShowAddSource, setEditShowAddSource] = useState<boolean>(false)

  // Available documents & chunks for evidence selection
  const [availableDocs, setAvailableDocs] = useState<DocumentListItem[]>([])
  const [loadingDocs, setLoadingDocs] = useState<boolean>(false)
  const [selectedDocId, setSelectedDocId] = useState<string>('')
  const [docChunks, setDocChunks] = useState<DocumentChunkInfo[]>([])
  const [loadingChunks, setLoadingChunks] = useState<boolean>(false)
  const [selectedChunkId, setSelectedChunkId] = useState<string>('')
  const [chunkFilterText, setChunkFilterText] = useState<string>('')

  // Attach Evidence State
  const [attachTab, setAttachTab] = useState<'DOCUMENT' | 'WEBSITE'>('DOCUMENT')
  const [attachExcerpt, setAttachExcerpt] = useState<string>('')
  const [attachUrl, setAttachUrl] = useState<string>('')
  const [attachUrlTitle, setAttachUrlTitle] = useState<string>('')
  const [attachUrlExcerpt, setAttachUrlExcerpt] = useState<string>('')

  // Batch Selection State
  const [selectedNoteIds, setSelectedNoteIds] = useState<string[]>([])
  const [batchSubmitting, setBatchSubmitting] = useState<boolean>(false)

  // Sequence ref to cancel / ignore out-of-order network responses
  const fetchSeqRef = useRef<number>(0)

  // Ref for the inner evidence selector drawer inside Create/Edit modal
  const innerSelectorRef = useRef<HTMLDivElement>(null)

  // Collapse inner evidence selector on outside click or Escape
  useEffect(() => {
    if (!createShowAddSource && !editShowAddSource) return

    const handlePointerDown = (e: MouseEvent | TouchEvent) => {
      if (innerSelectorRef.current && innerSelectorRef.current.contains(e.target as Node)) {
        return
      }
      if (createShowAddSource) setCreateShowAddSource(false)
      if (editShowAddSource) setEditShowAddSource(false)
    }

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation()
        e.preventDefault()
        if (createShowAddSource) setCreateShowAddSource(false)
        if (editShowAddSource) setEditShowAddSource(false)
      }
    }

    document.addEventListener('mousedown', handlePointerDown)
    document.addEventListener('keydown', handleKeyDown, true)

    return () => {
      document.removeEventListener('mousedown', handlePointerDown)
      document.removeEventListener('keydown', handleKeyDown, true)
    }
  }, [createShowAddSource, editShowAddSource])

  // Escape key handler for main modals when inner selector is not open
  useEffect(() => {
    const handleModalKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (createShowAddSource || editShowAddSource) return
        if (showAttachModal) {
          setShowAttachModal(false)
        } else if (showEditModal) {
          setShowEditModal(false)
        } else if (showCreateModal) {
          setShowCreateModal(false)
        } else if (batchModalResult) {
          setBatchModalResult(null)
        }
      }
    }

    window.addEventListener('keydown', handleModalKeyDown)
    return () => window.removeEventListener('keydown', handleModalKeyDown)
  }, [createShowAddSource, editShowAddSource, showAttachModal, showEditModal, showCreateModal, batchModalResult])

  // Debounce searchInput into search state (300ms)
  useEffect(() => {
    const timer = setTimeout(() => {
      setSearch(searchInput.trim())
    }, 300)
    return () => clearTimeout(timer)
  }, [searchInput])

  const existingCompanies = useMemo(() => {
    return Array.from(new Set(notes.map((n) => n.company_name).filter(Boolean))).sort()
  }, [notes])

  const loadNotes = async () => {
    const seq = ++fetchSeqRef.current
    try {
      setLoading(true)
      setLoadError(null)
      setIsPermissionDenied(false)
      const data = await fetchCompanyNotes(token, {
        company_name: (filterCompany !== 'ALL' ? filterCompany : search) || undefined,
        approval_status: (filterStatus !== 'ALL' && filterStatus !== 'ARCHIVED' && filterStatus !== 'SUPERSEDED') ? filterStatus : undefined,
        visibility: filterVisibility !== 'ALL' ? filterVisibility : undefined,
        workspace_id: (filterWorkspaceOnly && workspaceId) ? workspaceId : undefined,
      })
      if (seq !== fetchSeqRef.current) return // Stale request ignored
      setNotes(data)
      if (selectedNote) {
        const refreshed = data.find((n) => n.id === selectedNote.id)
        if (refreshed) setSelectedNote(refreshed)
      }
    } catch (err: any) {
      if (seq !== fetchSeqRef.current) return
      const is403 = err.status === 403 || err.message?.includes('403') || err.message?.toLowerCase().includes('forbidden') || err.message?.toLowerCase().includes('permission')
      if (is403) {
        setIsPermissionDenied(true)
      }
      setLoadError(err.message || 'Failed to load company notes')
      onToast(err.message || 'Failed to load company notes', 'error')
    } finally {
      if (seq === fetchSeqRef.current) {
        setLoading(false)
      }
    }
  }

  useEffect(() => {
    loadNotes()
  }, [search, filterStatus, filterVisibility, filterCompany, filterWorkspaceOnly])

  const isValidHttpUrl = (str: string) => {
    try {
      const url = new URL(str.trim())
      return url.protocol === 'http:' || url.protocol === 'https:'
    } catch (_) {
      return false
    }
  }

  const loadAvailableDocs = async () => {
    try {
      setLoadingDocs(true)
      const docs = await listDocuments(token)
      setAvailableDocs(docs)
      if (docs.length > 0) {
        setSelectedDocId(docs[0].id)
        loadChunksForDoc(docs[0].id)
      }
    } catch (err: any) {
      onToast(err.message || 'Failed to load documents for evidence attachment', 'error')
    } finally {
      setLoadingDocs(false)
    }
  }

  const loadChunksForDoc = async (docId: string) => {
    if (!docId) {
      setDocChunks([])
      setSelectedChunkId('')
      return
    }
    try {
      setLoadingChunks(true)
      setSelectedChunkId('')
      const chunks = await fetchDocumentChunks(token, docId)
      setDocChunks(chunks)
      if (chunks.length > 0) {
        setSelectedChunkId(chunks[0].id)
      }
    } catch (err: any) {
      onToast(err.message || 'Failed to load document chunks', 'error')
      setDocChunks([])
    } finally {
      setLoadingChunks(false)
    }
  }

  const handleCreateNote = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!createCompany.trim() || !createStatement.trim()) {
      onToast('Company name and observation statement are required', 'error')
      return
    }
    try {
      const created = await createCompanyNote(token, {
        company_name: createCompany.trim(),
        statement: createStatement.trim(),
        observation_type: createObsType,
        visibility: createVisibility,
        workspace_id: createVisibility === 'WORKSPACE' && workspaceId ? workspaceId : undefined,
        document_chunk_ids: createStagedDocs.map((d) => d.id),
        website_sources: createStagedWebsites.map((w) => ({
          url: w.url,
          title: w.title?.trim() || undefined,
          user_excerpt: w.user_excerpt?.trim() || undefined,
        })),
      })
      onToast('Company note draft created successfully!', 'success')
      setShowCreateModal(false)
      setCreateCompany('')
      setCreateStatement('')
      setCreateObsType('SUPPORTED')
      setCreateVisibility('PRIVATE')
      setCreateStagedDocs([])
      setCreateStagedWebsites([])
      setCreateShowAddSource(false)
      await loadNotes()
      setSelectedNote(created)
    } catch (err: any) {
      onToast(err.message || 'Failed to create company note', 'error')
    }
  }

  const handleUpdateNote = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!selectedNote) return
    try {
      const updated = await updateCompanyNote(token, selectedNote.id, {
        company_name: editCompany.trim(),
        statement: editStatement.trim(),
        observation_type: editObsType,
      })
      onToast('Note updated successfully', 'success')
      setShowEditModal(false)
      setSelectedNote(updated)
      await loadNotes()
    } catch (err: any) {
      onToast(err.message || 'Failed to update note', 'error')
    }
  }

  const handleAttachEvidence = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!selectedNote) return

    if (attachTab === 'DOCUMENT') {
      if (!selectedChunkId) {
        onToast('Please select a document chunk to attach', 'error')
        return
      }
      if (selectedNote.evidence.some((ev) => ev.document_chunk_id === selectedChunkId)) {
        onToast('This document chunk is already attached to this note', 'error')
        return
      }
      try {
        const updated = await attachNoteEvidence(token, selectedNote.id, {
          document_chunk_id: selectedChunkId,
          custom_excerpt: attachExcerpt.trim() || undefined,
        })
        onToast('Source document evidence attached successfully!', 'success')
        setShowAttachModal(false)
        setAttachExcerpt('')
        setSelectedChunkId('')
        setSelectedNote(updated)
        await loadNotes()
      } catch (err: any) {
        onToast(err.message || 'Failed to attach evidence (authorization check failed or chunk not found)', 'error')
      }
    } else {
      // Website source
      const url = attachUrl.trim()
      if (!url) {
        onToast('Website URL is required', 'error')
        return
      }
      if (!isValidHttpUrl(url)) {
        onToast('Invalid URL. Only HTTP and HTTPS URLs are allowed.', 'error')
        return
      }
      if (selectedNote.evidence.some((ev) => ev.source_url && ev.source_url.toLowerCase() === url.toLowerCase())) {
        onToast('This website source URL is already attached to this note', 'error')
        return
      }
      try {
        const updated = await attachNoteEvidence(token, selectedNote.id, {
          source_url: url,
          source_title: attachUrlTitle.trim() || undefined,
          custom_excerpt: attachUrlExcerpt.trim() || undefined,
        })
        onToast('Website source registered as NOT_VERIFIED citation!', 'success')
        setShowAttachModal(false)
        setAttachUrl('')
        setAttachUrlTitle('')
        setAttachUrlExcerpt('')
        setSelectedNote(updated)
        await loadNotes()
      } catch (err: any) {
        onToast(err.message || 'Failed to attach website source', 'error')
      }
    }
  }

  const handleAttachEvidenceInEdit = async () => {
    if (!selectedNote) return

    if (attachTab === 'DOCUMENT') {
      if (!selectedChunkId) {
        onToast('Please select a document chunk to attach', 'error')
        return
      }
      if (selectedNote.evidence.some((ev) => ev.document_chunk_id === selectedChunkId)) {
        onToast('This document chunk is already attached to this note', 'error')
        return
      }
      try {
        const updated = await attachNoteEvidence(token, selectedNote.id, {
          document_chunk_id: selectedChunkId,
        })
        onToast('Source document evidence attached successfully!', 'success')
        setSelectedNote(updated)
        setEditShowAddSource(false)
        setSelectedChunkId('')
        await loadNotes()
      } catch (err: any) {
        onToast(err.message || 'Failed to attach evidence (authorization check failed or chunk not found)', 'error')
      }
    } else {
      const url = attachUrl.trim()
      if (!url) {
        onToast('Website URL is required', 'error')
        return
      }
      if (!isValidHttpUrl(url)) {
        onToast('Invalid URL. Only HTTP and HTTPS URLs are allowed.', 'error')
        return
      }
      if (selectedNote.evidence.some((ev) => ev.source_url && ev.source_url.toLowerCase() === url.toLowerCase())) {
        onToast('This website source URL is already attached to this note', 'error')
        return
      }
      try {
        const updated = await attachNoteEvidence(token, selectedNote.id, {
          source_url: url,
          source_title: attachUrlTitle.trim() || undefined,
          custom_excerpt: attachUrlExcerpt.trim() || undefined,
        })
        onToast('Website source registered as NOT_VERIFIED citation!', 'success')
        setSelectedNote(updated)
        setEditShowAddSource(false)
        setAttachUrl('')
        setAttachUrlTitle('')
        setAttachUrlExcerpt('')
        await loadNotes()
      } catch (err: any) {
        onToast(err.message || 'Failed to attach website source', 'error')
      }
    }
  }

  const handleRemoveEvidence = async (evidenceId: string) => {
    if (!selectedNote) return
    try {
      const updated = await removeNoteEvidence(token, selectedNote.id, evidenceId)
      onToast('Evidence reference removed', 'success')
      setSelectedNote(updated)
      await loadNotes()
    } catch (err: any) {
      onToast(err.message || 'Failed to remove evidence', 'error')
    }
  }

  const handleSubmitForReview = async (noteId: string) => {
    try {
      const updated = await submitCompanyNote(token, noteId, {
        target_audience: 'SHARED',
      })
      onToast('Submitted note for publication review!', 'success')
      setSelectedNote(updated)
      await loadNotes()
    } catch (err: any) {
      onToast(err.message || 'Failed to submit note for review', 'error')
    }
  }

  const handleReviewAction = async (noteId: string, action: 'APPROVE' | 'REJECT' | 'REVISE') => {
    try {
      const updated = await reviewCompanyNote(token, noteId, { action })
      onToast(`Note ${action.toLowerCase()}d successfully!`, 'success')
      setSelectedNote(updated)
      await loadNotes()
    } catch (err: any) {
      onToast(err.message || `Failed to ${action.toLowerCase()} note`, 'error')
    }
  }

  const handleArchiveNote = async (noteId: string) => {
    if (!window.confirm('Are you sure you want to archive this company note? It will be removed from shared organizational retrieval.')) {
      return
    }
    try {
      const updated = await archiveCompanyNote(token, noteId)
      onToast('Note archived successfully', 'success')
      setSelectedNote(updated)
      await loadNotes()
    } catch (err: any) {
      onToast(err.message || 'Failed to archive note', 'error')
    }
  }

  const handleShareToWorkspace = async (noteId: string) => {
    if (!workspaceId) {
      onToast('No active workspace selected to share with', 'error')
      return
    }
    try {
      const updated = await shareNoteToWorkspace(token, noteId, workspaceId)
      onToast('Note shared with current workspace!', 'success')
      setSelectedNote(updated)
      await loadNotes()
    } catch (err: any) {
      onToast(err.message || 'Failed to share note to workspace', 'error')
    }
  }

  const handleBatchPublish = async () => {
    if (selectedNoteIds.length === 0) return
    try {
      setBatchSubmitting(true)
      const res = await batchPublishCompanyNotes(token, {
        note_ids: selectedNoteIds,
        target_audience: 'SHARED',
      })
      setBatchModalResult(res)
      onToast(`Batch completed: ${res.published_count} published, ${res.failed_count} failed`, 'success')
      setSelectedNoteIds([])
      await loadNotes()
    } catch (err: any) {
      onToast(err.message || 'Batch publication failed', 'error')
    } finally {
      setBatchSubmitting(false)
    }
  }

  const toggleSelectNote = (id: string) => {
    setSelectedNoteIds((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]
    )
  }

  const selectAllFiltered = () => {
    if (selectedNoteIds.length === filteredNotes.length) {
      setSelectedNoteIds([])
    } else {
      setSelectedNoteIds(filteredNotes.map((n) => n.id))
    }
  }

  const filteredNotes = notes.filter((n) => {
    if (search) {
      const q = search.toLowerCase()
      const matchStmt = (n.statement || '').toLowerCase().includes(q)
      const matchComp = (n.company_name || '').toLowerCase().includes(q)
      const matchAuth = (n.author_email || '').toLowerCase().includes(q)
      if (!matchStmt && !matchComp && !matchAuth) return false
    }
    if (filterObsType !== 'ALL' && n.observation_type !== filterObsType) return false
    if (filterCompany !== 'ALL' && n.company_name !== filterCompany) return false
    if (filterVisibility !== 'ALL' && n.visibility !== filterVisibility) return false

    if (filterStatus === 'ARCHIVED') {
      return n.status === 'ARCHIVED'
    } else if (filterStatus === 'SUPERSEDED') {
      return n.status === 'SUPERSEDED'
    } else if (filterStatus === 'APPROVED') {
      return n.status === 'ACTIVE' && n.approval_status === 'APPROVED'
    } else if (filterStatus === 'PENDING') {
      return n.status === 'ACTIVE' && n.approval_status === 'PENDING'
    } else if (filterStatus === 'DRAFT') {
      return n.status === 'ACTIVE' && n.approval_status === 'DRAFT'
    } else if (filterStatus === 'REJECTED') {
      return n.status === 'ACTIVE' && n.approval_status === 'REJECTED'
    }
    return true
  })

  // Badges & styling helpers
  const getObsBadge = (type: string) => {
    switch (type) {
      case 'SUPPORTED':
        return { label: 'Directly Supported', color: '#047857', bg: '#ecfdf5', border: '#a7f3d0' }
      case 'INFERRED':
        return { label: 'Inferred / Deduced', color: '#b45309', bg: '#fffbeb', border: '#fde68a' }
      case 'UNVERIFIED':
      default:
        return { label: 'Unverified Assertion', color: '#b91c1c', bg: '#fef2f2', border: '#fecaca' }
    }
  }

  const getStatusBadge = (approvalStatus: string, status?: string) => {
    if (status === 'ARCHIVED') {
      return { label: 'Archived', color: '#4b5563', bg: '#f3f4f6', border: '#e5e7eb' }
    }
    if (status === 'SUPERSEDED') {
      return { label: 'Superseded', color: '#9a3412', bg: '#ffedd5', border: '#fed7aa' }
    }
    if (status && status !== 'ACTIVE') {
      return { label: status, color: '#4b5563', bg: '#f3f4f6', border: '#e5e7eb' }
    }
    switch (approvalStatus) {
      case 'APPROVED':
        return { label: 'Approved', color: '#047857', bg: '#d1fae5', border: '#a7f3d0' }
      case 'PENDING':
        return { label: 'Pending Review', color: '#b45309', bg: '#fef3c7', border: '#fde68a' }
      case 'REJECTED':
        return { label: 'Rejected', color: '#b91c1c', bg: '#fee2e2', border: '#fecaca' }
      case 'DRAFT':
      default:
        return { label: 'Draft', color: '#4b5563', bg: '#f3f4f6', border: '#e5e7eb' }
    }
  }

  const getVisibilityBadge = (vis: string) => {
    switch (vis) {
      case 'SHARED':
        return { label: 'Shared Org-Wide', color: '#1d4ed8', bg: '#eff6ff', border: '#bfdbfe' }
      case 'WORKSPACE':
        return { label: 'Workspace', color: '#6d28d9', bg: '#f5f3ff', border: '#ddd6fe' }
      case 'PRIVATE':
      default:
        return { label: 'Private Draft', color: '#4b5563', bg: '#f3f4f6', border: '#e5e7eb' }
    }
  }

  return (
    <div style={{ flex: 1, minHeight: 0, height: '100%', display: 'flex', flexDirection: 'column', background: 'var(--color-bg, #f9fafb)' }}>
      {/* Top Header */}
      <header
        style={{
          padding: '20px 28px 16px',
          borderBottom: '1px solid var(--color-border, #e5e7eb)',
          background: 'var(--color-surface, #ffffff)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '16px',
        }}
      >
        <div>
          <h1 style={{ margin: 0, fontSize: '22px', fontWeight: 700, color: 'var(--color-ink, #111827)', fontFamily: 'var(--font-serif, inherit)' }}>
            Company Notes & Knowledge Review
          </h1>
          <p style={{ margin: '4px 0 0', fontSize: '13px', color: 'var(--color-ink-muted, #6b7280)' }}>
            Record client observations, attach authorized document citations, audit claims, and manage shared publication.
          </p>
        </div>

        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          {isAdmin && selectedNoteIds.length > 0 && (
            <button
              onClick={handleBatchPublish}
              disabled={batchSubmitting}
              style={{
                background: '#4f46e5',
                color: '#ffffff',
                border: 'none',
                padding: '8px 16px',
                borderRadius: '8px',
                fontSize: '13px',
                fontWeight: 600,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                boxShadow: '0 2px 4px rgba(79, 70, 229, 0.2)',
              }}
            >
              {batchSubmitting ? 'Publishing Batch…' : `Publish Selected (${selectedNoteIds.length})`}
            </button>
          )}

          <button
            onClick={() => setShowCreateModal(true)}
            style={{
              background: 'var(--color-teal, #0d9488)',
              color: '#ffffff',
              border: 'none',
              padding: '8px 16px',
              borderRadius: '8px',
              fontSize: '13px',
              fontWeight: 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              boxShadow: '0 2px 6px rgba(13, 148, 136, 0.25)',
            }}
          >
            <span>+</span> New Company Note
          </button>
        </div>
      </header>

      {/* Main Content Layout */}
      <div style={{ flex: 1, display: 'flex', minHeight: 0, overflow: 'hidden' }}>
        {/* Left Column: Notes List & Filters */}
        <div
          style={{
            flex: '1 1 52%',
            minWidth: '360px',
            borderRight: '1px solid var(--color-border, #e5e7eb)',
            display: 'flex',
            flexDirection: 'column',
            background: 'var(--color-surface, #ffffff)',
          }}
        >
          {/* Filter Bar */}
          <div style={{ padding: '12px 18px', borderBottom: '1px solid var(--color-border, #e5e7eb)', display: 'flex', flexDirection: 'column', gap: '8px', background: '#fafafa' }}>
            {/* Primary Filter Row */}
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' }}>
              <input
                type="text"
                placeholder="Search observations, companies..."
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value)}
                style={{
                  flex: '1 1 180px',
                  padding: '7px 11px',
                  borderRadius: '6px',
                  border: '1px solid #d1d5db',
                  fontSize: '13px',
                  background: '#ffffff',
                }}
              />

              <select
                value={filterStatus}
                onChange={(e) => setFilterStatus(e.target.value)}
                style={{ padding: '7px 10px', borderRadius: '6px', border: '1px solid #d1d5db', fontSize: '12px', background: '#ffffff', fontWeight: 500 }}
              >
                <option value="ALL">All Statuses</option>
                <option value="PENDING">Pending Review</option>
                <option value="APPROVED">Approved (Shared)</option>
                <option value="DRAFT">Draft</option>
                <option value="REJECTED">Rejected</option>
                <option value="SUPERSEDED">Superseded</option>
                <option value="ARCHIVED">Archived</option>
              </select>

              <select
                value={filterCompany}
                onChange={(e) => setFilterCompany(e.target.value)}
                style={{ padding: '7px 10px', borderRadius: '6px', border: '1px solid #d1d5db', fontSize: '12px', background: '#ffffff', maxWidth: '160px' }}
                title="Filter by authorized company"
              >
                <option value="ALL">All Companies ({existingCompanies.length})</option>
                {existingCompanies.map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>

              <button
                type="button"
                onClick={() => setShowMoreFilters(!showMoreFilters)}
                style={{
                  padding: '6px 10px',
                  borderRadius: '6px',
                  border: '1px solid #d1d5db',
                  background: showMoreFilters ? '#e0f2fe' : '#ffffff',
                  color: showMoreFilters ? '#0369a1' : '#4b5563',
                  fontSize: '12px',
                  cursor: 'pointer',
                  fontWeight: 500,
                }}
              >
                {showMoreFilters ? 'Fewer Filters ▲' : 'More Filters ▼'}
              </button>

              {isAdmin && filteredNotes.length > 0 && (
                <button
                  onClick={selectAllFiltered}
                  style={{
                    background: '#f3f4f6',
                    border: '1px solid #d1d5db',
                    padding: '6px 10px',
                    borderRadius: '6px',
                    fontSize: '12px',
                    fontWeight: 500,
                    cursor: 'pointer',
                    color: '#374151',
                  }}
                >
                  {selectedNoteIds.length === filteredNotes.length ? 'Deselect All' : 'Select All'}
                </button>
              )}
            </div>

            {/* Compact Secondary Filters Row */}
            {showMoreFilters && (
              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center', paddingTop: '4px', borderTop: '1px dashed #e5e7eb' }}>
                <select
                  value={filterVisibility}
                  onChange={(e) => setFilterVisibility(e.target.value)}
                  style={{ padding: '5px 8px', borderRadius: '5px', border: '1px solid #d1d5db', fontSize: '11px', background: '#ffffff' }}
                >
                  <option value="ALL">All Visibilities</option>
                  <option value="PRIVATE">Private to Me</option>
                  <option value="WORKSPACE">Workspace</option>
                  <option value="SHARED">Shared Org</option>
                </select>

                <select
                  value={filterObsType}
                  onChange={(e) => setFilterObsType(e.target.value)}
                  style={{ padding: '5px 8px', borderRadius: '5px', border: '1px solid #d1d5db', fontSize: '11px', background: '#ffffff' }}
                >
                  <option value="ALL">All Evidence Types</option>
                  <option value="SUPPORTED">Directly Supported</option>
                  <option value="INFERRED">Inferred</option>
                  <option value="UNVERIFIED">Unverified</option>
                </select>

                {workspaceId && (
                  <label style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '11px', color: '#4b5563', cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      checked={filterWorkspaceOnly}
                      onChange={(e) => setFilterWorkspaceOnly(e.target.checked)}
                    />
                    Current Workspace Only
                  </label>
                )}
              </div>
            )}
          </div>

          {/* Notes List */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '12px 16px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {isPermissionDenied && (
              <div style={{ padding: '14px 16px', background: '#FEF2F2', border: '1px solid #FECACA', borderRadius: '8px', color: '#991B1B' }}>
                <div style={{ fontWeight: 600, fontSize: '13px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span>🔒</span> Permission Denied
                </div>
                <div style={{ fontSize: '12px', marginTop: '4px', color: '#B91C1C' }}>
                  You do not have authorization to view Company Notes in this scope or department. Contact an administrator to request access.
                </div>
              </div>
            )}

            {loadError && !isPermissionDenied && (
              <div style={{ padding: '12px 16px', background: '#FFFBEB', border: '1px solid #FDE68A', borderRadius: '8px', color: '#92400E', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <div style={{ fontWeight: 600, fontSize: '12px' }}>Failed to load notes</div>
                  <div style={{ fontSize: '11px' }}>{loadError}</div>
                </div>
                <button
                  onClick={loadNotes}
                  style={{ padding: '4px 10px', fontSize: '11px', borderRadius: '4px', border: '1px solid #D97706', background: '#FFF', color: '#B45309', cursor: 'pointer' }}
                >
                  Retry
                </button>
              </div>
            )}

            {loading ? (
              <div style={{ padding: '32px', textAlign: 'center', color: '#9ca3af', fontSize: '14px' }}>Loading observations…</div>
            ) : filteredNotes.length === 0 ? (
              <div style={{ padding: '48px 24px', textAlign: 'center', color: '#9ca3af' }}>
                <div style={{ fontSize: '32px', marginBottom: '8px' }}>📝</div>
                <div style={{ fontSize: '15px', fontWeight: 600, color: '#374151' }}>No observations found</div>
                <div style={{ fontSize: '13px', marginTop: '4px' }}>Create an observation draft or adjust your search filters.</div>
              </div>
            ) : (
              filteredNotes.map((note) => {
                const obsBadge = getObsBadge(note.observation_type)
                const statBadge = getStatusBadge(note.approval_status, note.status)
                const visBadge = getVisibilityBadge(note.visibility)
                const isSelected = selectedNote?.id === note.id
                const isChecked = selectedNoteIds.includes(note.id)

                return (
                  <div
                    key={note.id}
                    onClick={() => setSelectedNote(note)}
                    style={{
                      padding: '13px 15px',
                      borderRadius: '9px',
                      border: isSelected ? '2px solid #0d9488' : '1px solid #e5e7eb',
                      background: isSelected ? '#f0fdfa' : '#ffffff',
                      cursor: 'pointer',
                      transition: 'all 0.15s ease',
                      boxShadow: isSelected ? '0 4px 12px rgba(13, 148, 136, 0.1)' : '0 1px 3px rgba(0,0,0,0.03)',
                      display: 'flex',
                      gap: '12px',
                    }}
                  >
                    {isAdmin && (
                      <div
                        onClick={(e) => {
                          e.stopPropagation()
                          toggleSelectNote(note.id)
                        }}
                        style={{ paddingTop: '2px' }}
                      >
                        <input
                          type="checkbox"
                          checked={isChecked}
                          onChange={() => {}}
                          style={{ cursor: 'pointer', width: '15px', height: '15px' }}
                        />
                      </div>
                    )}

                    <div style={{ flex: 1, minWidth: 0 }}>
                      {/* Primary Heading: The observation claim */}
                      <div style={{ fontSize: '14px', fontWeight: 600, color: '#111827', lineHeight: '1.4', marginBottom: '5px', wordBreak: 'break-word' }}>
                        {note.statement || 'Client observation draft'}
                      </div>

                      {/* Secondary Metadata: Company Name & Author */}
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap', marginBottom: '8px', fontSize: '12px', color: '#4b5563' }}>
                        {note.company_name && (
                          <span style={{ fontWeight: 600, color: '#1f2937', display: 'flex', alignItems: 'center', gap: '3px' }}>
                            🏢 {note.company_name}
                          </span>
                        )}
                        {note.author_email && (
                          <span style={{ color: '#6b7280' }}>by {note.author_email}</span>
                        )}
                        <span style={{ color: '#9ca3af' }}>•</span>
                        <span style={{ color: '#6b7280' }}>{new Date(note.created_at).toLocaleDateString()}</span>
                      </div>

                      {/* Badges & Source Count */}
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '6px' }}>
                        <div style={{ display: 'flex', gap: '5px', alignItems: 'center', flexWrap: 'wrap' }}>
                          <span style={{ fontSize: '11px', padding: '2px 7px', borderRadius: '4px', background: statBadge.bg, color: statBadge.color, border: `1px solid ${statBadge.border}`, fontWeight: 600 }}>
                            {statBadge.label}
                          </span>
                          <span style={{ fontSize: '11px', padding: '2px 7px', borderRadius: '4px', background: visBadge.bg, color: visBadge.color, border: `1px solid ${visBadge.border}`, fontWeight: 600 }}>
                            {visBadge.label}
                          </span>
                          <span style={{ fontSize: '11px', padding: '2px 7px', borderRadius: '4px', background: obsBadge.bg, color: obsBadge.color, border: `1px solid ${obsBadge.border}`, fontWeight: 600 }}>
                            {obsBadge.label}
                          </span>
                        </div>

                        <span
                          style={{
                            fontSize: '11px',
                            fontWeight: 500,
                            color: note.evidence_count > 0 ? '#4b5563' : '#b45309',
                            background: note.evidence_count > 0 ? '#f3f4f6' : '#fef3c7',
                            padding: '2px 7px',
                            borderRadius: '4px',
                            border: `1px solid ${note.evidence_count > 0 ? '#e5e7eb' : '#fde68a'}`,
                          }}
                          title={note.evidence_count === 0 ? 'Observation has no evidence attached. At least 1 citation is required for publication review.' : `${note.evidence_count} grounding sources`}
                        >
                          📎 {note.evidence_count} source{note.evidence_count !== 1 ? 's' : ''}
                        </span>
                      </div>
                    </div>
                  </div>
                )
              })
            )}
          </div>
        </div>

        {/* Right Column: Note Evidence Review & Detail Pane */}
        <div
          style={{
            flex: '1 1 45%',
            minWidth: '380px',
            background: 'var(--color-bg, #f9fafb)',
            overflowY: 'auto',
            padding: '24px 28px',
          }}
        >
          {selectedNote ? (
            <div>
              {/* 1. Observation & Company Summary */}
              <div
                style={{
                  background: '#ffffff',
                  borderRadius: '12px',
                  border: '1px solid #e5e7eb',
                  padding: '20px',
                  marginBottom: '18px',
                  boxShadow: '0 1px 4px rgba(0,0,0,0.04)',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' }}>
                  <div style={{ flex: 1, minWidth: 0, marginRight: '12px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                      <span style={{ fontSize: '13px', fontWeight: 700, color: '#0f766e', display: 'flex', alignItems: 'center', gap: '4px' }}>
                        🏢 {selectedNote.company_name || 'Observation'}
                      </span>
                      {selectedNote.department_code && (
                        <span style={{ fontSize: '11px', background: '#f1f5f9', color: '#475569', padding: '1px 6px', borderRadius: '4px' }}>
                          Dept: {selectedNote.department_code}
                        </span>
                      )}
                    </div>
                    {/* Primary Heading: The observation claim */}
                    <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 700, color: '#111827', lineHeight: '1.4' }}>
                      {selectedNote.statement}
                    </h2>
                  </div>

                  <div style={{ display: 'flex', gap: '6px', flexShrink: 0 }}>
                    {selectedNote.status === 'ACTIVE' && selectedNote.approval_status !== 'APPROVED' && (
                      <button
                        onClick={() => {
                          setEditCompany(selectedNote.company_name)
                          setEditStatement(selectedNote.statement)
                          setEditObsType(selectedNote.observation_type as any)
                          setShowEditModal(true)
                        }}
                        style={{
                          background: '#f3f4f6',
                          border: '1px solid #d1d5db',
                          borderRadius: '6px',
                          padding: '6px 12px',
                          fontSize: '12px',
                          fontWeight: 500,
                          cursor: 'pointer',
                        }}
                      >
                        Edit Note
                      </button>
                    )}
                  </div>
                </div>

                {/* Status Banners for Archived / Superseded items */}
                {selectedNote.status === 'ARCHIVED' && (
                  <div style={{ background: '#f3f4f6', border: '1px solid #e5e7eb', borderRadius: '6px', padding: '10px 14px', marginBottom: '14px', fontSize: '12px', color: '#4b5563' }}>
                    📦 <strong>Archived Observation:</strong> This note has been withdrawn and excluded from active retrieval.
                  </div>
                )}
                {selectedNote.status === 'SUPERSEDED' && (
                  <div style={{ background: '#fff7ed', border: '1px solid #ffedd5', borderRadius: '6px', padding: '10px 14px', marginBottom: '14px', fontSize: '12px', color: '#9a3412' }}>
                    🔄 <strong>Superseded:</strong> This observation was replaced by an updated note and is no longer retrieved for answers.
                  </div>
                )}

                {/* 2. Approval, Visibility and Lifecycle State Pills */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '10px', fontSize: '12px', marginBottom: '16px' }}>
                  <div style={{ background: '#f8fafc', padding: '8px 12px', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
                    <div style={{ color: '#64748b', fontSize: '10px', textTransform: 'uppercase', fontWeight: 600 }}>Review Status</div>
                    <div style={{ fontWeight: 600, color: '#0f172a', marginTop: '2px' }}>
                      {getStatusBadge(selectedNote.approval_status, selectedNote.status).label}
                    </div>
                  </div>

                  <div style={{ background: '#f8fafc', padding: '8px 12px', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
                    <div style={{ color: '#64748b', fontSize: '10px', textTransform: 'uppercase', fontWeight: 600 }}>Publication Scope</div>
                    <div style={{ fontWeight: 600, color: '#0f172a', marginTop: '2px' }}>
                      {getVisibilityBadge(selectedNote.visibility).label}
                    </div>
                  </div>

                  <div style={{ background: '#f8fafc', padding: '8px 12px', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
                    <div style={{ color: '#64748b', fontSize: '10px', textTransform: 'uppercase', fontWeight: 600 }}>Support Level</div>
                    <div style={{ fontWeight: 600, color: '#0f172a', marginTop: '2px' }}>
                      {getObsBadge(selectedNote.observation_type).label}
                    </div>
                  </div>

                  <div style={{ background: '#f8fafc', padding: '8px 12px', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
                    <div style={{ color: '#64748b', fontSize: '10px', textTransform: 'uppercase', fontWeight: 600 }}>Grounding Evidence</div>
                    <div style={{ fontWeight: 600, color: selectedNote.evidence_count > 0 ? '#0f172a' : '#b45309', marginTop: '2px' }}>
                      {selectedNote.evidence_count} source citation{selectedNote.evidence_count !== 1 ? 's' : ''}
                    </div>
                  </div>
                </div>

                {/* 4. Source Provenance, Timestamps and Author/Reviewer info */}
                <div style={{ background: '#f9fafb', borderRadius: '8px', border: '1px solid #f3f4f6', padding: '12px 14px', fontSize: '12px', color: '#4b5563', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
                    <span>
                      <strong>Author:</strong> {selectedNote.author_email || 'Author'}
                    </span>
                    <span>
                      <strong>Created:</strong> {new Date(selectedNote.created_at).toLocaleString()}
                    </span>
                  </div>
                  {selectedNote.approved_at && (
                    <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px', color: '#065f46', paddingTop: '4px', borderTop: '1px dashed #e5e7eb' }}>
                      <span>
                        <strong>Approved by:</strong> {selectedNote.approved_by_email || 'Admin'}
                      </span>
                      <span>
                        <strong>Approved at:</strong> {new Date(selectedNote.approved_at).toLocaleString()}
                      </span>
                    </div>
                  )}
                  {selectedNote.canonical_key && (
                    <div style={{ fontSize: '11px', color: '#9ca3af', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', paddingTop: '2px' }}>
                      Ref Key: <code>{selectedNote.canonical_key}</code>
                    </div>
                  )}
                </div>

                {/* 5. Review Decision Controls & 6. Secondary Actions */}
                <div style={{ marginTop: '16px', paddingTop: '16px', borderTop: '1px solid #f3f4f6' }}>
                  {/* Actions Row */}
                  <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
                    {/* DRAFT Submission */}
                    {selectedNote.status === 'ACTIVE' && selectedNote.approval_status === 'DRAFT' && (
                      <button
                        onClick={() => handleSubmitForReview(selectedNote.id)}
                        disabled={selectedNote.evidence.filter((ev) => ev.source_type === 'DOCUMENT' || Boolean(ev.document_chunk_id)).length === 0}
                        style={{
                          background: selectedNote.evidence.filter((ev) => ev.source_type === 'DOCUMENT' || Boolean(ev.document_chunk_id)).length > 0 ? '#0d9488' : '#9ca3af',
                          color: '#ffffff',
                          border: 'none',
                          padding: '8px 14px',
                          borderRadius: '6px',
                          fontSize: '12px',
                          fontWeight: 600,
                          cursor: selectedNote.evidence.filter((ev) => ev.source_type === 'DOCUMENT' || Boolean(ev.document_chunk_id)).length > 0 ? 'pointer' : 'not-allowed',
                          opacity: selectedNote.evidence.filter((ev) => ev.source_type === 'DOCUMENT' || Boolean(ev.document_chunk_id)).length > 0 ? 1 : 0.7,
                        }}
                        title={
                          selectedNote.evidence.filter((ev) => ev.source_type === 'DOCUMENT' || Boolean(ev.document_chunk_id)).length === 0
                            ? 'Submission for publication review requires at least 1 verified document citation'
                            : 'Submit note for admin publication review'
                        }
                      >
                        Submit for Publication Review
                      </button>
                    )}

                    {/* Admin Review Decision Controls */}
                    {isAdmin && selectedNote.status === 'ACTIVE' && selectedNote.approval_status === 'PENDING' && (
                      <>
                        <button
                          onClick={() => handleReviewAction(selectedNote.id, 'APPROVE')}
                          disabled={selectedNote.evidence.filter((ev) => ev.source_type === 'DOCUMENT' || Boolean(ev.document_chunk_id)).length === 0}
                          style={{
                            background: selectedNote.evidence.filter((ev) => ev.source_type === 'DOCUMENT' || Boolean(ev.document_chunk_id)).length > 0 ? '#059669' : '#9ca3af',
                            color: '#ffffff',
                            border: 'none',
                            padding: '8px 14px',
                            borderRadius: '6px',
                            fontSize: '12px',
                            fontWeight: 600,
                            cursor: selectedNote.evidence.filter((ev) => ev.source_type === 'DOCUMENT' || Boolean(ev.document_chunk_id)).length > 0 ? 'pointer' : 'not-allowed',
                            opacity: selectedNote.evidence.filter((ev) => ev.source_type === 'DOCUMENT' || Boolean(ev.document_chunk_id)).length > 0 ? 1 : 0.7,
                          }}
                          title={
                            selectedNote.evidence.filter((ev) => ev.source_type === 'DOCUMENT' || Boolean(ev.document_chunk_id)).length === 0
                              ? 'Cannot approve: observation requires at least 1 verified document citation'
                              : 'Approve note and publish to shared organizational knowledge'
                          }
                        >
                          ✓ Approve & Publish Shared
                        </button>
                        <button
                          onClick={() => handleReviewAction(selectedNote.id, 'REJECT')}
                          style={{
                            background: '#dc2626',
                            color: '#ffffff',
                            border: 'none',
                            padding: '8px 14px',
                            borderRadius: '6px',
                            fontSize: '12px',
                            fontWeight: 600,
                            cursor: 'pointer',
                          }}
                        >
                          ✗ Reject
                        </button>
                        <button
                          onClick={() => handleReviewAction(selectedNote.id, 'REVISE')}
                          style={{
                            background: '#6b7280',
                            color: '#ffffff',
                            border: 'none',
                            padding: '8px 14px',
                            borderRadius: '6px',
                            fontSize: '12px',
                            fontWeight: 600,
                            cursor: 'pointer',
                          }}
                        >
                          ↺ Return for Revision
                        </button>
                      </>
                    )}

                    {/* Secondary Action: Workspace Sharing */}
                    {selectedNote.status === 'ACTIVE' && selectedNote.visibility === 'PRIVATE' && workspaceId && (
                      <button
                        onClick={() => handleShareToWorkspace(selectedNote.id)}
                        style={{
                          background: '#f8fafc',
                          color: '#0f766e',
                          border: '1px solid #0d9488',
                          padding: '7px 12px',
                          borderRadius: '6px',
                          fontSize: '12px',
                          fontWeight: 600,
                          cursor: 'pointer',
                        }}
                      >
                        👥 Share to Current Workspace
                      </button>
                    )}

                    {/* Secondary Action: Archive */}
                    {selectedNote.status !== 'ARCHIVED' && (
                      <button
                        onClick={() => handleArchiveNote(selectedNote.id)}
                        style={{
                          background: '#ffffff',
                          color: '#6b7280',
                          border: '1px solid #d1d5db',
                          padding: '7px 12px',
                          borderRadius: '6px',
                          fontSize: '12px',
                          fontWeight: 500,
                          cursor: 'pointer',
                          marginLeft: 'auto',
                        }}
                        title="Archive this note (withdraws from active search and retrieval)"
                      >
                        📦 Archive Note
                      </button>
                    )}
                  </div>

                  {/* Evidence Requirement Alert if 0 citations */}
                  {selectedNote.evidence.length === 0 && selectedNote.status === 'ACTIVE' && (
                    <div style={{ marginTop: '10px', fontSize: '11px', color: '#b45309', background: '#fef3c7', padding: '6px 10px', borderRadius: '4px', border: '1px solid #fde68a' }}>
                      ⚠️ <strong>Evidence Rule:</strong> Company notes require at least 1 verified document citation before publication review or approval.
                    </div>
                  )}

                  {/* Evidence Alert if only website citations exist */}
                  {selectedNote.evidence.length > 0 && selectedNote.evidence.filter((ev) => ev.source_type === 'DOCUMENT' || Boolean(ev.document_chunk_id)).length === 0 && selectedNote.status === 'ACTIVE' && (
                    <div style={{ marginTop: '10px', fontSize: '11px', color: '#b45309', background: '#fef3c7', padding: '6px 10px', borderRadius: '4px', border: '1px solid #fde68a' }}>
                      ⚠️ <strong>Publication Rule:</strong> Note references {selectedNote.evidence.length} website source(s) without an authorized document citation. Website sources are stored as unverified citations in this version. Shared publication requires at least 1 verified document source.
                    </div>
                  )}
                </div>
              </div>

              {/* 3. Supporting Evidence and Excerpts */}
              <div
                style={{
                  background: '#ffffff',
                  borderRadius: '12px',
                  border: '1px solid #e5e7eb',
                  padding: '20px',
                  boxShadow: '0 1px 4px rgba(0,0,0,0.04)',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                  <div>
                    <h3 style={{ margin: 0, fontSize: '15px', fontWeight: 700, color: '#111827' }}>
                      Supporting Evidence Sources ({selectedNote.evidence.length})
                    </h3>
                    <p style={{ margin: '2px 0 0 0', fontSize: '12px', color: '#6b7280' }}>
                      {selectedNote.evidence.filter((ev) => ev.source_type === 'DOCUMENT' || Boolean(ev.document_chunk_id)).length} document citation(s) • {selectedNote.evidence.filter((ev) => ev.source_type === 'WEBSITE' || Boolean(ev.source_url)).length} website source(s)
                    </p>
                  </div>

                  {selectedNote.status === 'ACTIVE' && selectedNote.approval_status !== 'APPROVED' && (
                    <button
                      onClick={() => {
                        setShowAttachModal(true)
                        if (availableDocs.length === 0) loadAvailableDocs()
                      }}
                      style={{
                        background: '#f3f4f6',
                        border: '1px solid #d1d5db',
                        padding: '6px 12px',
                        borderRadius: '6px',
                        fontSize: '12px',
                        fontWeight: 600,
                        cursor: 'pointer',
                        color: '#374151',
                      }}
                    >
                      + Attach Source
                    </button>
                  )}
                </div>

                {selectedNote.evidence.length === 0 ? (
                  <div style={{ padding: '24px', textAlign: 'center', background: '#fffbeb', borderRadius: '8px', border: '1px solid #fef3c7' }}>
                    <div style={{ fontSize: '13px', fontWeight: 600, color: '#b45309' }}>No Supporting Sources Attached</div>
                    <div style={{ fontSize: '12px', color: '#92400e', marginTop: '4px' }}>
                      Click "+ Attach Source" to attach an authorized document chunk or register a website source.
                    </div>
                  </div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                    {selectedNote.evidence.map((ev) => {
                      const isWeb = ev.source_type === 'WEBSITE' || Boolean(ev.source_url)
                      return (
                        <div
                          key={ev.id}
                          style={{
                            background: isWeb ? '#f8fafc' : '#f9fafb',
                            borderRadius: '8px',
                            border: `1px solid ${isWeb ? '#e2e8f0' : '#e5e7eb'}`,
                            padding: '14px',
                          }}
                        >
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '8px' }}>
                            <div style={{ flex: 1, marginRight: '12px' }}>
                              {isWeb ? (
                                <div>
                                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                                    <span style={{ fontSize: '14px' }}>🌐</span>
                                    <a
                                      href={ev.source_url || '#'}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      style={{ fontSize: '13px', fontWeight: 600, color: '#0284c7', textDecoration: 'none', wordBreak: 'break-all' }}
                                    >
                                      {ev.source_title ? `${ev.source_title} (${ev.source_url})` : ev.source_url} ↗
                                    </a>
                                    <span
                                      style={{
                                        padding: '2px 6px',
                                        borderRadius: '4px',
                                        fontSize: '10px',
                                        fontWeight: 600,
                                        background: ev.verification_status === 'VERIFIED' ? '#dcfce7' : '#fef3c7',
                                        color: ev.verification_status === 'VERIFIED' ? '#166534' : '#b45309',
                                        border: `1px solid ${ev.verification_status === 'VERIFIED' ? '#bbf7d0' : '#fde68a'}`,
                                      }}
                                    >
                                      {ev.verification_status === 'VERIFIED' ? 'Verified Page' : 'Not Verified Web Source'}
                                    </span>
                                  </div>
                                  <div style={{ fontSize: '11px', color: '#64748b', marginTop: '4px' }}>
                                    Registered: {new Date(ev.created_at).toLocaleString()} • Web retrieval & verification deferred to future version
                                  </div>
                                </div>
                              ) : (
                                <div>
                                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                    <span style={{ fontSize: '14px' }}>📄</span>
                                    <span style={{ fontSize: '13px', fontWeight: 600, color: '#111827' }}>
                                      {ev.original_file_name || ev.file_name || 'Authorized Document'}
                                    </span>
                                    <span style={{ padding: '2px 6px', borderRadius: '4px', fontSize: '10px', fontWeight: 600, background: '#dcfce7', color: '#166534', border: '1px solid #bbf7d0' }}>
                                      Authorized Doc Chunk
                                    </span>
                                  </div>
                                  <div style={{ fontSize: '11px', color: '#6b7280', marginTop: '2px' }}>
                                    {ev.page_number ? `Page ${ev.page_number}` : ''}
                                    {ev.section_title ? ` • Section: ${ev.section_title}` : ''}
                                    {ev.section_ordinal ? ` (${ev.section_ordinal})` : ''}
                                    {' • Attached: ' + new Date(ev.created_at).toLocaleDateString()}
                                  </div>
                                </div>
                              )}
                            </div>

                            {selectedNote.status === 'ACTIVE' && selectedNote.approval_status !== 'APPROVED' && (
                              <button
                                onClick={() => handleRemoveEvidence(ev.id)}
                                style={{
                                  background: 'transparent',
                                  border: 'none',
                                  color: '#dc2626',
                                  fontSize: '12px',
                                  cursor: 'pointer',
                                  padding: '2px 6px',
                                }}
                                title="Remove source from note"
                              >
                                Remove
                              </button>
                            )}
                          </div>

                          {isWeb && ev.user_excerpt && (
                            <div style={{ marginTop: '8px' }}>
                              <div style={{ fontSize: '10px', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', marginBottom: '2px' }}>
                                User-Provided Supporting Passage
                              </div>
                              <div
                                style={{
                                  background: '#ffffff',
                                  padding: '10px 12px',
                                  borderRadius: '6px',
                                  border: '1px solid #e2e8f0',
                                  fontSize: '12px',
                                  color: '#334155',
                                  fontStyle: 'italic',
                                  lineHeight: '1.4',
                                }}
                              >
                                "{ev.user_excerpt}"
                              </div>
                            </div>
                          )}

                          {!isWeb && ev.excerpt && (
                            <div
                              style={{
                                background: '#ffffff',
                                padding: '10px 12px',
                                borderRadius: '6px',
                                border: '1px solid #e5e7eb',
                                fontSize: '12px',
                                color: '#374151',
                                lineHeight: '1.4',
                                fontFamily: 'monospace',
                              }}
                            >
                              "{ev.excerpt}"
                            </div>
                          )}
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
            </div>
          ) : (
            /* Compact Empty State with clear guidance and quick summary */
            <div
              style={{
                background: '#ffffff',
                borderRadius: '12px',
                border: '1px solid #e5e7eb',
                padding: '32px 24px',
                textAlign: 'center',
                boxShadow: '0 1px 4px rgba(0,0,0,0.04)',
              }}
            >
              <div style={{ fontSize: '32px', marginBottom: '10px' }}>📋</div>
              <h3 style={{ margin: '0 0 6px 0', fontSize: '16px', fontWeight: 700, color: '#111827' }}>
                Knowledge Review Inspector
              </h3>
              <p style={{ margin: '0 auto 20px auto', fontSize: '13px', color: '#6b7280', maxWidth: '380px', lineHeight: '1.5' }}>
                Select an observation on the left to inspect grounded citations, verify claims against authorized documents, or take review decisions.
              </p>

              {/* Compact Quick Stats */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px', maxWidth: '420px', margin: '0 auto 20px auto' }}>
                <div style={{ background: '#f8fafc', padding: '10px', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                  <div style={{ fontSize: '18px', fontWeight: 700, color: '#0f172a' }}>{notes.length}</div>
                  <div style={{ fontSize: '11px', color: '#64748b' }}>Total Notes</div>
                </div>
                <div style={{ background: '#fffbeb', padding: '10px', borderRadius: '8px', border: '1px solid #fde68a' }}>
                  <div style={{ fontSize: '18px', fontWeight: 700, color: '#b45309' }}>
                    {notes.filter((n) => n.status === 'ACTIVE' && n.approval_status === 'PENDING').length}
                  </div>
                  <div style={{ fontSize: '11px', color: '#92400e' }}>Pending Review</div>
                </div>
                <div style={{ background: '#ecfdf5', padding: '10px', borderRadius: '8px', border: '1px solid #a7f3d0' }}>
                  <div style={{ fontSize: '18px', fontWeight: 700, color: '#047857' }}>
                    {notes.filter((n) => n.status === 'ACTIVE' && n.approval_status === 'APPROVED').length}
                  </div>
                  <div style={{ fontSize: '11px', color: '#065f46' }}>Approved Shared</div>
                </div>
              </div>

              <button
                onClick={() => {
                  setCreateCompany('')
                  setCreateStatement('')
                  setShowCreateModal(true)
                }}
                style={{
                  background: '#0d9488',
                  color: '#ffffff',
                  border: 'none',
                  padding: '9px 18px',
                  borderRadius: '6px',
                  fontSize: '13px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '6px',
                }}
              >
                + Create Company Note
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Modal: Create Company Note */}
      {showCreateModal && (
        <div style={modalOverlayStyle}>
          <div style={modalBoxStyle}>
            <div style={modalHeaderStyle}>
              <h3 style={{ margin: 0, fontSize: '17px', fontWeight: 700, color: '#111827' }}>
                Create Company Note Observation
              </h3>
              <button
                type="button"
                onClick={() => setShowCreateModal(false)}
                style={closeBtnStyle}
                title="Close dialog (Esc)"
              >
                ✕
              </button>
            </div>
            <form onSubmit={handleCreateNote} style={modalFormStyle}>
              <div style={modalBodyStyle}>
                <div style={{ marginBottom: '14px' }}>
                  <label style={labelStyle}>Company / Client Name *</label>
                  <input
                    type="text"
                    list="authorized-companies-datalist"
                    required
                    placeholder="e.g. Acme Corp, TechVenture Partners"
                    value={createCompany}
                    onChange={(e) => setCreateCompany(e.target.value)}
                    style={inputStyle}
                  />
                  <datalist id="authorized-companies-datalist">
                    {existingCompanies.map((c) => (
                      <option key={c} value={c} />
                    ))}
                  </datalist>
                  {existingCompanies.length > 0 && (
                    <div style={{ display: 'flex', gap: '6px', alignItems: 'center', flexWrap: 'wrap', marginTop: '6px' }}>
                      <span style={{ fontSize: '11px', color: '#64748b' }}>Authorized companies:</span>
                      {existingCompanies.slice(0, 6).map((c) => (
                        <button
                          type="button"
                          key={c}
                          onClick={() => setCreateCompany(c)}
                          style={{
                            padding: '2px 8px',
                            fontSize: '11px',
                            borderRadius: '4px',
                            border: '1px solid #cbd5e1',
                            background: createCompany === c ? '#e0f2fe' : '#f1f5f9',
                            color: createCompany === c ? '#0369a1' : '#334155',
                            cursor: 'pointer',
                          }}
                        >
                          {c}
                        </button>
                      ))}
                    </div>
                  )}
                </div>

                <div style={{ marginBottom: '14px' }}>
                  <label style={labelStyle}>Observation / Business Statement *</label>
                  <textarea
                    required
                    rows={4}
                    placeholder="State the observation (e.g. Acme Corp plans 50-seat enterprise deployment in Q4)..."
                    value={createStatement}
                    onChange={(e) => setCreateStatement(e.target.value)}
                    style={{ ...inputStyle, resize: 'vertical' }}
                  />
                </div>

                <div style={{ marginBottom: '14px' }}>
                  <label style={labelStyle}>Evidence Support Level *</label>
                  <div style={{ display: 'flex', gap: '8px' }}>
                    {(['SUPPORTED', 'INFERRED', 'UNVERIFIED'] as const).map((type) => {
                      const badge = getObsBadge(type)
                      const active = createObsType === type
                      return (
                        <button
                          key={type}
                          type="button"
                          onClick={() => setCreateObsType(type)}
                          style={{
                            flex: 1,
                            padding: '8px 10px',
                            borderRadius: '6px',
                            border: `2px solid ${active ? badge.color : '#e5e7eb'}`,
                            background: active ? badge.bg : '#ffffff',
                            color: active ? badge.color : '#4b5563',
                            fontSize: '12px',
                            fontWeight: 600,
                            cursor: 'pointer',
                          }}
                        >
                          {badge.label}
                        </button>
                      )
                    })}
                  </div>
                </div>

                {/* Staged Supporting Evidence Section in Create Modal */}
                <div style={{ marginBottom: '18px', padding: '12px', background: '#f8fafc', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                    <label style={{ ...labelStyle, marginBottom: 0 }}>
                      Supporting Sources ({createStagedDocs.length + createStagedWebsites.length})
                    </label>
                    <div style={{ display: 'flex', gap: '6px' }}>
                      <button
                        type="button"
                        onClick={() => {
                          setCreateShowAddSource(true)
                          setAttachTab('DOCUMENT')
                          if (availableDocs.length === 0) loadAvailableDocs()
                        }}
                        style={{ padding: '3px 8px', fontSize: '11px', borderRadius: '4px', border: '1px solid #cbd5e1', background: '#fff', cursor: 'pointer', color: '#0f766e', fontWeight: 600 }}
                      >
                        + Add Document Chunk
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          setCreateShowAddSource(true)
                          setAttachTab('WEBSITE')
                        }}
                        style={{ padding: '3px 8px', fontSize: '11px', borderRadius: '4px', border: '1px solid #cbd5e1', background: '#fff', cursor: 'pointer', color: '#0369a1', fontWeight: 600 }}
                      >
                        + Add Website URL
                      </button>
                    </div>
                  </div>

                  {createStagedDocs.length === 0 && createStagedWebsites.length === 0 ? (
                    <div style={{ fontSize: '11px', color: '#94a3b8', fontStyle: 'italic', padding: '4px 0' }}>
                      Optional: You can stage document chunks or website URLs now, or attach them after creating the draft.
                    </div>
                  ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginBottom: '8px' }}>
                      {createStagedDocs.map((doc, idx) => (
                        <div key={idx} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: '#fff', padding: '6px 10px', borderRadius: '6px', border: '1px solid #e2e8f0', fontSize: '11px' }}>
                          <div style={{ minWidth: 0, marginRight: '8px' }}>
                            <strong>📄 {doc.fileName}</strong> {doc.page ? `(Page ${doc.page})` : ''} — <em>"{formatSafeExcerpt(doc.excerpt, 90)}"</em>
                          </div>
                          <button
                            type="button"
                            onClick={() => setCreateStagedDocs((prev) => prev.filter((_, i) => i !== idx))}
                            style={{ background: 'none', border: 'none', color: '#dc2626', cursor: 'pointer', fontWeight: 700, flexShrink: 0 }}
                            title="Remove staged document"
                          >
                            ×
                          </button>
                        </div>
                      ))}
                      {createStagedWebsites.map((web, idx) => (
                        <div key={idx} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: '#fff', padding: '6px 10px', borderRadius: '6px', border: '1px solid #e2e8f0', fontSize: '11px' }}>
                          <div style={{ minWidth: 0, marginRight: '8px' }}>
                            <strong>🌐 {web.title || web.url}</strong> <span style={{ color: '#b45309' }}>[Not Verified]</span> {web.user_excerpt ? `— "${formatSafeExcerpt(web.user_excerpt, 70)}"` : ''}
                          </div>
                          <button
                            type="button"
                            onClick={() => setCreateStagedWebsites((prev) => prev.filter((_, i) => i !== idx))}
                            style={{ background: 'none', border: 'none', color: '#dc2626', cursor: 'pointer', fontWeight: 700, flexShrink: 0 }}
                            title="Remove staged website"
                          >
                            ×
                          </button>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Inline Drawer for adding source to staged list */}
                  {createShowAddSource && (
                    <div
                      ref={innerSelectorRef}
                      onClick={(e) => e.stopPropagation()}
                      style={{
                        marginTop: '10px',
                        padding: '12px',
                        background: '#ffffff',
                        borderRadius: '8px',
                        border: '1px solid #cbd5e1',
                        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.05)',
                        maxHeight: '340px',
                        overflowY: 'auto',
                        overflowX: 'hidden',
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                        <span style={{ fontSize: '12px', fontWeight: 600, color: '#1e293b' }}>
                          {attachTab === 'DOCUMENT' ? 'Select Document Chunk to Stage' : 'Register Website Source to Stage'}
                        </span>
                        <button
                          type="button"
                          onClick={() => setCreateShowAddSource(false)}
                          style={{
                            background: 'none',
                            border: 'none',
                            color: '#64748b',
                            fontSize: '12px',
                            fontWeight: 600,
                            cursor: 'pointer',
                            padding: '2px 6px',
                            borderRadius: '4px',
                          }}
                          title="Collapse selector (Esc)"
                        >
                          ✕ Close
                        </button>
                      </div>

                      {attachTab === 'DOCUMENT' ? (
                        <div>
                          <select
                            value={selectedDocId}
                            onChange={(e) => {
                              setSelectedDocId(e.target.value)
                              loadChunksForDoc(e.target.value)
                            }}
                            style={{ ...inputStyle, marginBottom: '8px', fontSize: '12px' }}
                          >
                            <option value="">-- Choose an authorized document --</option>
                            {availableDocs.map((d) => (
                              <option key={d.id} value={d.id}>
                                {getDocumentDisplayName(d)} ({d.department_code || 'General'})
                              </option>
                            ))}
                          </select>

                          {loadingChunks ? (
                            <div style={{ fontSize: '11px', color: '#64748b', padding: '8px 0' }}>Loading document chunks…</div>
                          ) : docChunks.length > 0 ? (
                            <div style={{ maxHeight: '160px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '6px', marginBottom: '8px' }}>
                              {docChunks.map((chunk) => {
                                const isSelected = selectedChunkId === chunk.id
                                const selectedDoc = availableDocs.find((d) => d.id === selectedDocId)
                                const locationLabel = [
                                  chunk.page_number != null ? `Page ${chunk.page_number}` : null,
                                  chunk.section_title ? `Section: ${chunk.section_title}` : null,
                                ].filter(Boolean).join(' • ')

                                return (
                                  <div
                                    key={chunk.id}
                                    onClick={() => setSelectedChunkId(chunk.id)}
                                    style={{
                                      padding: '8px 10px',
                                      borderRadius: '6px',
                                      border: `1px solid ${isSelected ? '#0d9488' : '#e2e8f0'}`,
                                      background: isSelected ? '#f0fdfa' : '#ffffff',
                                      cursor: 'pointer',
                                      transition: 'all 0.15s ease',
                                      display: 'flex',
                                      flexDirection: 'column',
                                      gap: '4px',
                                    }}
                                  >
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', minWidth: 0 }}>
                                        <span style={{ fontSize: '12px' }}>📄</span>
                                        <span style={{ fontWeight: 600, fontSize: '12px', color: isSelected ? '#0f766e' : '#1e293b', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                          {getDocumentDisplayName(selectedDoc)}
                                        </span>
                                        {locationLabel && (
                                          <span style={{ padding: '1px 6px', background: '#f1f5f9', color: '#475569', borderRadius: '4px', fontSize: '10px', fontWeight: 500, flexShrink: 0 }}>
                                            {locationLabel}
                                          </span>
                                        )}
                                      </div>
                                      {isSelected && (
                                        <span style={{ fontSize: '11px', color: '#0d9488', fontWeight: 700, flexShrink: 0 }}>
                                          ✓ Selected
                                        </span>
                                      )}
                                    </div>
                                    <div style={{ color: '#475569', fontSize: '11px', lineHeight: '1.35', fontStyle: 'italic', wordBreak: 'break-word' }}>
                                      "{formatSafeExcerpt(chunk.excerpt, 130)}"
                                    </div>
                                  </div>
                                )
                              })}
                            </div>
                          ) : selectedDocId ? (
                            <div style={{ fontSize: '11px', color: '#64748b', marginBottom: '8px' }}>No chunks available for this document.</div>
                          ) : null}

                          <button
                            type="button"
                            disabled={!selectedChunkId}
                            onClick={() => {
                              const chunk = docChunks.find((c) => c.id === selectedChunkId)
                              const doc = availableDocs.find((d) => d.id === selectedDocId)
                              if (chunk) {
                                if (createStagedDocs.some((d) => d.id === chunk.id)) {
                                  onToast('This document chunk is already staged', 'error')
                                  return
                                }
                                setCreateStagedDocs((prev) => [
                                  ...prev,
                                  {
                                    id: chunk.id,
                                    fileName: getDocumentDisplayName(doc),
                                    excerpt: chunk.excerpt,
                                    page: chunk.page_number,
                                  },
                                ])
                                setCreateShowAddSource(false)
                                setSelectedChunkId('')
                              }
                            }}
                            style={{ ...btnPrimaryStyle, padding: '4px 10px', fontSize: '11px', opacity: selectedChunkId ? 1 : 0.6 }}
                          >
                            + Stage Selected Chunk
                          </button>
                        </div>
                      ) : (
                        <div>
                          <input
                            type="url"
                            placeholder="Website URL (e.g. https://example.com/press-release)"
                            value={attachUrl}
                            onChange={(e) => setAttachUrl(e.target.value)}
                            style={{ ...inputStyle, marginBottom: '6px', fontSize: '12px' }}
                          />
                          <input
                            type="text"
                            placeholder="Source Title / Attribution (Optional)"
                            value={attachUrlTitle}
                            onChange={(e) => setAttachUrlTitle(e.target.value)}
                            style={{ ...inputStyle, marginBottom: '6px', fontSize: '12px' }}
                          />
                          <textarea
                            rows={2}
                            placeholder="User-provided supporting passage (Optional)..."
                            value={attachUrlExcerpt}
                            onChange={(e) => setAttachUrlExcerpt(e.target.value)}
                            style={{ ...inputStyle, marginBottom: '8px', fontSize: '12px', resize: 'vertical' }}
                          />
                          <button
                            type="button"
                            disabled={!attachUrl.trim()}
                            onClick={() => {
                              const cleanUrl = attachUrl.trim()
                              if (!isValidHttpUrl(cleanUrl)) {
                                onToast('Invalid URL. Must start with http:// or https://', 'error')
                                return
                              }
                              if (createStagedWebsites.some((w) => w.url.toLowerCase() === cleanUrl.toLowerCase())) {
                                onToast('This website source URL is already staged', 'error')
                                return
                              }
                              setCreateStagedWebsites((prev) => [
                                ...prev,
                                {
                                  url: cleanUrl,
                                  title: attachUrlTitle.trim() || undefined,
                                  user_excerpt: attachUrlExcerpt.trim() || undefined,
                                },
                              ])
                              setAttachUrl('')
                              setAttachUrlTitle('')
                              setAttachUrlExcerpt('')
                              setCreateShowAddSource(false)
                            }}
                            style={{ ...btnPrimaryStyle, padding: '4px 10px', fontSize: '11px', opacity: attachUrl.trim() ? 1 : 0.6 }}
                          >
                            + Stage Website Source
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                <div style={{ marginBottom: '20px' }}>
                  <label style={labelStyle}>Draft Privacy & Scope</label>
                  <div style={{ display: 'flex', gap: '8px' }}>
                    <button
                      type="button"
                      onClick={() => setCreateVisibility('PRIVATE')}
                      style={{
                        flex: 1,
                        padding: '8px 10px',
                        borderRadius: '6px',
                        border: `2px solid ${createVisibility === 'PRIVATE' ? '#0d9488' : '#e5e7eb'}`,
                        background: createVisibility === 'PRIVATE' ? '#f0fdfa' : '#ffffff',
                        color: createVisibility === 'PRIVATE' ? '#0d9488' : '#4b5563',
                        fontSize: '12px',
                        fontWeight: 600,
                        cursor: 'pointer',
                      }}
                    >
                      🔒 Private Draft (Author Only)
                    </button>
                    <button
                      type="button"
                      onClick={() => setCreateVisibility('WORKSPACE')}
                      style={{
                        flex: 1,
                        padding: '8px 10px',
                        borderRadius: '6px',
                        border: `2px solid ${createVisibility === 'WORKSPACE' ? '#0d9488' : '#e5e7eb'}`,
                        background: createVisibility === 'WORKSPACE' ? '#f0fdfa' : '#ffffff',
                        color: createVisibility === 'WORKSPACE' ? '#0d9488' : '#4b5563',
                        fontSize: '12px',
                        fontWeight: 600,
                        cursor: 'pointer',
                      }}
                    >
                      👥 Workspace Draft
                    </button>
                  </div>
                </div>
              </div>

              <div style={modalFooterStyle}>
                <button
                  type="button"
                  onClick={() => setShowCreateModal(false)}
                  style={btnSecondaryStyle}
                >
                  Cancel
                </button>
                <button type="submit" style={btnPrimaryStyle}>
                  Create Note Draft
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Modal: Edit Company Note */}
      {showEditModal && selectedNote && (
        <div style={modalOverlayStyle}>
          <div style={modalBoxStyle}>
            <div style={modalHeaderStyle}>
              <h3 style={{ margin: 0, fontSize: '17px', fontWeight: 700, color: '#111827' }}>
                Edit Company Note
              </h3>
              <button
                type="button"
                onClick={() => setShowEditModal(false)}
                style={closeBtnStyle}
                title="Close dialog (Esc)"
              >
                ✕
              </button>
            </div>
            <form onSubmit={handleUpdateNote} style={modalFormStyle}>
              <div style={modalBodyStyle}>
                <div style={{ marginBottom: '14px' }}>
                  <label style={labelStyle}>Company Name</label>
                  <input
                    type="text"
                    list="authorized-companies-datalist"
                    required
                    value={editCompany}
                    onChange={(e) => setEditCompany(e.target.value)}
                    style={inputStyle}
                  />
                </div>

                <div style={{ marginBottom: '14px' }}>
                  <label style={labelStyle}>Observation Statement</label>
                  <textarea
                    required
                    rows={4}
                    value={editStatement}
                    onChange={(e) => setEditStatement(e.target.value)}
                    style={{ ...inputStyle, resize: 'vertical' }}
                  />
                </div>

                <div style={{ marginBottom: '14px' }}>
                  <label style={labelStyle}>Evidence Support Level</label>
                  <div style={{ display: 'flex', gap: '8px' }}>
                    {(['SUPPORTED', 'INFERRED', 'UNVERIFIED'] as const).map((type) => {
                      const badge = getObsBadge(type)
                      const active = editObsType === type
                      return (
                        <button
                          key={type}
                          type="button"
                          onClick={() => setEditObsType(type)}
                          style={{
                            flex: 1,
                            padding: '8px 10px',
                            borderRadius: '6px',
                            border: `2px solid ${active ? badge.color : '#e5e7eb'}`,
                            background: active ? badge.bg : '#ffffff',
                            color: active ? badge.color : '#4b5563',
                            fontSize: '12px',
                            fontWeight: 600,
                            cursor: 'pointer',
                          }}
                        >
                          {badge.label}
                        </button>
                      )
                    })}
                  </div>
                </div>

                {/* Supporting Evidence in Edit Modal */}
                <div style={{ marginBottom: '18px', padding: '12px', background: '#f8fafc', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                    <label style={{ ...labelStyle, marginBottom: 0 }}>
                      Attached Sources ({selectedNote.evidence.length})
                    </label>
                    <div style={{ display: 'flex', gap: '6px' }}>
                      <button
                        type="button"
                        onClick={() => {
                          setEditShowAddSource(true)
                          setAttachTab('DOCUMENT')
                          if (availableDocs.length === 0) loadAvailableDocs()
                        }}
                        style={{ padding: '3px 8px', fontSize: '11px', borderRadius: '4px', border: '1px solid #cbd5e1', background: '#fff', cursor: 'pointer', color: '#0f766e', fontWeight: 600 }}
                      >
                        + Add Document Chunk
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          setEditShowAddSource(true)
                          setAttachTab('WEBSITE')
                        }}
                        style={{ padding: '3px 8px', fontSize: '11px', borderRadius: '4px', border: '1px solid #cbd5e1', background: '#fff', cursor: 'pointer', color: '#0369a1', fontWeight: 600 }}
                      >
                        + Add Website URL
                      </button>
                    </div>
                  </div>

                  {selectedNote.evidence.length === 0 ? (
                    <div style={{ fontSize: '11px', color: '#94a3b8', fontStyle: 'italic', padding: '4px 0' }}>
                      No sources attached yet. Click "+ Add Document Chunk" or "+ Add Website URL" to link sources.
                    </div>
                  ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', maxHeight: '140px', overflowY: 'auto' }}>
                      {selectedNote.evidence.map((ev) => {
                        const isWeb = ev.source_type === 'WEBSITE' || Boolean(ev.source_url)
                        return (
                          <div key={ev.id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: '#fff', padding: '6px 10px', borderRadius: '6px', border: '1px solid #e2e8f0', fontSize: '11px' }}>
                            <div style={{ minWidth: 0, marginRight: '8px' }}>
                              {isWeb ? (
                                <span>🌐 <strong>{ev.source_title || ev.source_url}</strong> <span style={{ color: '#b45309' }}>[Not Verified]</span> {ev.user_excerpt ? `— "${formatSafeExcerpt(ev.user_excerpt, 70)}"` : ''}</span>
                              ) : (
                                <span>📄 <strong>{ev.original_file_name || ev.file_name || 'Document'}</strong> {ev.page_number ? `(Page ${ev.page_number})` : ''} {ev.section_title ? `• ${ev.section_title}` : ''} {ev.custom_excerpt ? `— "${formatSafeExcerpt(ev.custom_excerpt, 70)}"` : ''}</span>
                              )}
                            </div>
                            <button
                              type="button"
                              onClick={() => handleRemoveEvidence(ev.id)}
                              style={{ background: 'none', border: 'none', color: '#dc2626', cursor: 'pointer', fontSize: '11px', flexShrink: 0, fontWeight: 700 }}
                              title="Remove this source"
                            >
                              Remove
                            </button>
                          </div>
                        )
                      })}
                    </div>
                  )}

                  {/* Inline Drawer for adding source directly in Edit modal */}
                  {editShowAddSource && (
                    <div
                      ref={innerSelectorRef}
                      onClick={(e) => e.stopPropagation()}
                      style={{
                        marginTop: '10px',
                        padding: '12px',
                        background: '#ffffff',
                        borderRadius: '8px',
                        border: '1px solid #cbd5e1',
                        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.05)',
                        maxHeight: '340px',
                        overflowY: 'auto',
                        overflowX: 'hidden',
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                        <span style={{ fontSize: '12px', fontWeight: 600, color: '#1e293b' }}>
                          {attachTab === 'DOCUMENT' ? 'Select Document Chunk to Attach' : 'Register Website Source to Attach'}
                        </span>
                        <button
                          type="button"
                          onClick={() => setEditShowAddSource(false)}
                          style={{
                            background: 'none',
                            border: 'none',
                            color: '#64748b',
                            fontSize: '12px',
                            fontWeight: 600,
                            cursor: 'pointer',
                            padding: '2px 6px',
                            borderRadius: '4px',
                          }}
                          title="Collapse selector (Esc)"
                        >
                          ✕ Close
                        </button>
                      </div>

                      {attachTab === 'DOCUMENT' ? (
                        <div>
                          <select
                            value={selectedDocId}
                            onChange={(e) => {
                              setSelectedDocId(e.target.value)
                              loadChunksForDoc(e.target.value)
                            }}
                            style={{ ...inputStyle, marginBottom: '8px', fontSize: '12px' }}
                          >
                            <option value="">-- Choose an authorized document --</option>
                            {availableDocs.map((d) => (
                              <option key={d.id} value={d.id}>
                                {getDocumentDisplayName(d)} ({d.department_code || 'General'})
                              </option>
                            ))}
                          </select>

                          {loadingChunks ? (
                            <div style={{ fontSize: '11px', color: '#64748b', padding: '8px 0' }}>Loading document chunks…</div>
                          ) : docChunks.length > 0 ? (
                            <div style={{ maxHeight: '160px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '6px', marginBottom: '8px' }}>
                              {docChunks.map((chunk) => {
                                const isSelected = selectedChunkId === chunk.id
                                const selectedDoc = availableDocs.find((d) => d.id === selectedDocId)
                                const locationLabel = [
                                  chunk.page_number != null ? `Page ${chunk.page_number}` : null,
                                  chunk.section_title ? `Section: ${chunk.section_title}` : null,
                                ].filter(Boolean).join(' • ')

                                return (
                                  <div
                                    key={chunk.id}
                                    onClick={() => setSelectedChunkId(chunk.id)}
                                    style={{
                                      padding: '8px 10px',
                                      borderRadius: '6px',
                                      border: `1px solid ${isSelected ? '#0d9488' : '#e2e8f0'}`,
                                      background: isSelected ? '#f0fdfa' : '#ffffff',
                                      cursor: 'pointer',
                                      transition: 'all 0.15s ease',
                                      display: 'flex',
                                      flexDirection: 'column',
                                      gap: '4px',
                                    }}
                                  >
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', minWidth: 0 }}>
                                        <span style={{ fontSize: '12px' }}>📄</span>
                                        <span style={{ fontWeight: 600, fontSize: '12px', color: isSelected ? '#0f766e' : '#1e293b', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                          {getDocumentDisplayName(selectedDoc)}
                                        </span>
                                        {locationLabel && (
                                          <span style={{ padding: '1px 6px', background: '#f1f5f9', color: '#475569', borderRadius: '4px', fontSize: '10px', fontWeight: 500, flexShrink: 0 }}>
                                            {locationLabel}
                                          </span>
                                        )}
                                      </div>
                                      {isSelected && (
                                        <span style={{ fontSize: '11px', color: '#0d9488', fontWeight: 700, flexShrink: 0 }}>
                                          ✓ Selected
                                        </span>
                                      )}
                                    </div>
                                    <div style={{ color: '#475569', fontSize: '11px', lineHeight: '1.35', fontStyle: 'italic', wordBreak: 'break-word' }}>
                                      "{formatSafeExcerpt(chunk.excerpt, 130)}"
                                    </div>
                                  </div>
                                )
                              })}
                            </div>
                          ) : selectedDocId ? (
                            <div style={{ fontSize: '11px', color: '#64748b', marginBottom: '8px' }}>No chunks available for this document.</div>
                          ) : null}

                          <button
                            type="button"
                            disabled={!selectedChunkId}
                            onClick={handleAttachEvidenceInEdit}
                            style={{ ...btnPrimaryStyle, padding: '4px 10px', fontSize: '11px', opacity: selectedChunkId ? 1 : 0.6 }}
                          >
                            + Attach Selected Chunk
                          </button>
                        </div>
                      ) : (
                        <div>
                          <input
                            type="url"
                            placeholder="Website URL (e.g. https://example.com/press-release)"
                            value={attachUrl}
                            onChange={(e) => setAttachUrl(e.target.value)}
                            style={{ ...inputStyle, marginBottom: '6px', fontSize: '12px' }}
                          />
                          <input
                            type="text"
                            placeholder="Source Title / Attribution (Optional)"
                            value={attachUrlTitle}
                            onChange={(e) => setAttachUrlTitle(e.target.value)}
                            style={{ ...inputStyle, marginBottom: '6px', fontSize: '12px' }}
                          />
                          <textarea
                            rows={2}
                            placeholder="User-provided supporting passage (Optional)..."
                            value={attachUrlExcerpt}
                            onChange={(e) => setAttachUrlExcerpt(e.target.value)}
                            style={{ ...inputStyle, marginBottom: '8px', fontSize: '12px', resize: 'vertical' }}
                          />
                          <button
                            type="button"
                            disabled={!attachUrl.trim()}
                            onClick={handleAttachEvidenceInEdit}
                            style={{ ...btnPrimaryStyle, padding: '4px 10px', fontSize: '11px', opacity: attachUrl.trim() ? 1 : 0.6 }}
                          >
                            + Attach Website Source
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>

              <div style={modalFooterStyle}>
                <button
                  type="button"
                  onClick={() => setShowEditModal(false)}
                  style={btnSecondaryStyle}
                >
                  Cancel
                </button>
                <button type="submit" style={btnPrimaryStyle}>
                  Save Changes
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Modal: Attach Evidence (Document Selection or Website Registration) */}
      {showAttachModal && (
        <div style={modalOverlayStyle}>
          <div style={{ ...modalBoxStyle, maxWidth: '560px' }}>
            <div style={modalHeaderStyle}>
              <h3 style={{ margin: 0, fontSize: '17px', fontWeight: 700, color: '#111827' }}>
                Attach Supporting Evidence Source
              </h3>
              <button
                type="button"
                onClick={() => setShowAttachModal(false)}
                style={closeBtnStyle}
                title="Close dialog (Esc)"
              >
                ✕
              </button>
            </div>

            <form onSubmit={handleAttachEvidence} style={modalFormStyle}>
              <div style={modalBodyStyle}>
                {/* Source Type Selector Tabs */}
                <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
                  <button
                    type="button"
                    onClick={() => setAttachTab('DOCUMENT')}
                    style={{
                      flex: 1,
                      padding: '8px 12px',
                      borderRadius: '6px',
                      border: `2px solid ${attachTab === 'DOCUMENT' ? '#0d9488' : '#e5e7eb'}`,
                      background: attachTab === 'DOCUMENT' ? '#f0fdfa' : '#ffffff',
                      color: attachTab === 'DOCUMENT' ? '#0d9488' : '#4b5563',
                      fontSize: '12px',
                      fontWeight: 600,
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: '6px',
                    }}
                  >
                    <span>📄</span> Authorized Document Chunk
                  </button>
                  <button
                    type="button"
                    onClick={() => setAttachTab('WEBSITE')}
                    style={{
                      flex: 1,
                      padding: '8px 12px',
                      borderRadius: '6px',
                      border: `2px solid ${attachTab === 'WEBSITE' ? '#0284c7' : '#e5e7eb'}`,
                      background: attachTab === 'WEBSITE' ? '#f0f9ff' : '#ffffff',
                      color: attachTab === 'WEBSITE' ? '#0284c7' : '#4b5563',
                      fontSize: '12px',
                      fontWeight: 600,
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: '6px',
                    }}
                  >
                    <span>🌐</span> Website Source (Unverified)
                  </button>
                </div>

                {attachTab === 'DOCUMENT' ? (
                  <div>
                    <div style={{ marginBottom: '14px' }}>
                      <label style={labelStyle}>Select Authorized Document *</label>
                      {loadingDocs ? (
                        <div style={{ fontSize: '12px', color: '#64748b', padding: '8px' }}>Loading authorized documents…</div>
                      ) : (
                        <select
                          value={selectedDocId}
                          onChange={(e) => {
                            setSelectedDocId(e.target.value)
                            loadChunksForDoc(e.target.value)
                          }}
                          required
                          style={inputStyle}
                        >
                          <option value="">-- Choose an authorized document --</option>
                          {availableDocs.map((doc) => (
                            <option key={doc.id} value={doc.id}>
                              {getDocumentDisplayName(doc)} ({doc.department_code || doc.department_name || 'General'} • {doc.status})
                            </option>
                          ))}
                        </select>
                      )}
                    </div>

                    {selectedDocId && (
                      <div style={{ marginBottom: '14px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                          <label style={{ ...labelStyle, marginBottom: 0 }}>Select Chunk or Section *</label>
                          <input
                            type="text"
                            placeholder="Filter passages…"
                            value={chunkFilterText}
                            onChange={(e) => setChunkFilterText(e.target.value)}
                            style={{ padding: '3px 8px', borderRadius: '4px', border: '1px solid #d1d5db', fontSize: '11px', width: '140px' }}
                          />
                        </div>

                        {loadingChunks ? (
                          <div style={{ fontSize: '12px', color: '#64748b', padding: '12px', textAlign: 'center' }}>Loading document chunks…</div>
                        ) : docChunks.length === 0 ? (
                          <div style={{ fontSize: '12px', color: '#64748b', padding: '12px', background: '#f9fafb', borderRadius: '6px', textAlign: 'center' }}>
                            No chunks found for this document.
                          </div>
                        ) : (
                          <div style={{ maxHeight: '180px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '6px', border: '1px solid #e5e7eb', borderRadius: '6px', padding: '6px' }}>
                            {docChunks
                              .filter((chunk) => {
                                if (!chunkFilterText.trim()) return true
                                const q = chunkFilterText.toLowerCase()
                                return (
                                  (chunk.section_title || '').toLowerCase().includes(q) ||
                                  (chunk.excerpt || '').toLowerCase().includes(q)
                                )
                              })
                              .map((chunk) => {
                                const isSelected = selectedChunkId === chunk.id
                                const selectedDoc = availableDocs.find((d) => d.id === selectedDocId)
                                const locationLabel = [
                                  chunk.page_number != null ? `Page ${chunk.page_number}` : null,
                                  chunk.section_title ? `Section: ${chunk.section_title}` : null,
                                ].filter(Boolean).join(' • ')

                                return (
                                  <div
                                    key={chunk.id}
                                    onClick={() => setSelectedChunkId(chunk.id)}
                                    style={{
                                      padding: '8px 10px',
                                      borderRadius: '6px',
                                      border: `1px solid ${isSelected ? '#0d9488' : '#e5e7eb'}`,
                                      background: isSelected ? '#f0fdfa' : '#ffffff',
                                      cursor: 'pointer',
                                      transition: 'all 0.15s ease',
                                      display: 'flex',
                                      flexDirection: 'column',
                                      gap: '4px',
                                    }}
                                  >
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', minWidth: 0 }}>
                                        <span style={{ fontSize: '12px' }}>📄</span>
                                        <span style={{ fontWeight: 600, fontSize: '12px', color: isSelected ? '#0f766e' : '#111827', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                          {getDocumentDisplayName(selectedDoc)}
                                        </span>
                                        {locationLabel && (
                                          <span style={{ padding: '1px 6px', background: '#f1f5f9', color: '#475569', borderRadius: '4px', fontSize: '10px', fontWeight: 500, flexShrink: 0 }}>
                                            {locationLabel}
                                          </span>
                                        )}
                                      </div>
                                      {isSelected && (
                                        <span style={{ fontSize: '11px', color: '#0d9488', fontWeight: 700, flexShrink: 0 }}>
                                          ✓ Selected
                                        </span>
                                      )}
                                    </div>
                                    <div style={{ fontSize: '11px', color: '#4b5563', lineHeight: '1.35', fontStyle: 'italic', wordBreak: 'break-word' }}>
                                      "{formatSafeExcerpt(chunk.excerpt, 140)}"
                                    </div>
                                  </div>
                                )
                              })}
                          </div>
                        )}
                      </div>
                    )}

                    <div style={{ marginBottom: '18px' }}>
                      <label style={labelStyle}>Specific Note or Custom Excerpt (Optional)</label>
                      <textarea
                        rows={2}
                        placeholder="Leave blank to use chunk excerpt, or specify a customized passage..."
                        value={attachExcerpt}
                        onChange={(e) => setAttachExcerpt(e.target.value)}
                        style={{ ...inputStyle, resize: 'vertical' }}
                      />
                    </div>
                  </div>
                ) : (
                  <div>
                    <div style={{ marginBottom: '14px' }}>
                      <label style={labelStyle}>Website URL *</label>
                      <input
                        type="url"
                        required
                        placeholder="https://example.com/company-update"
                        value={attachUrl}
                        onChange={(e) => setAttachUrl(e.target.value)}
                        style={inputStyle}
                      />
                      <div style={{ fontSize: '11px', color: '#64748b', marginTop: '4px' }}>
                        Allowed schemes: <code>http://</code> or <code>https://</code>.
                      </div>
                    </div>

                    <div style={{ marginBottom: '14px' }}>
                      <label style={labelStyle}>Source Title / Attribution (Optional)</label>
                      <input
                        type="text"
                        placeholder="e.g. Acme Corp Official Blog, Industry Announcement"
                        value={attachUrlTitle}
                        onChange={(e) => setAttachUrlTitle(e.target.value)}
                        style={inputStyle}
                      />
                    </div>

                    <div style={{ marginBottom: '14px' }}>
                      <label style={labelStyle}>User-Provided Supporting Passage (Optional)</label>
                      <textarea
                        rows={3}
                        placeholder="Enter the specific text or quote from the website that supports this claim..."
                        value={attachUrlExcerpt}
                        onChange={(e) => setAttachUrlExcerpt(e.target.value)}
                        style={{ ...inputStyle, resize: 'vertical' }}
                      />
                    </div>

                    <div style={{ background: '#fffbeb', border: '1px solid #fde68a', borderRadius: '6px', padding: '10px 12px', fontSize: '11px', color: '#92400e', marginBottom: '18px', lineHeight: '1.4' }}>
                      ℹ️ <strong>Website Source Notice:</strong> In this version, website citations are registered as <code>NOT_VERIFIED</code> without web crawling or automated fetching. Note publication still requires at least 1 verified document source.
                    </div>
                  </div>
                )}
              </div>

              <div style={modalFooterStyle}>
                <button
                  type="button"
                  onClick={() => setShowAttachModal(false)}
                  style={btnSecondaryStyle}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={attachTab === 'DOCUMENT' ? !selectedChunkId : !attachUrl.trim()}
                  style={{
                    ...btnPrimaryStyle,
                    opacity: (attachTab === 'DOCUMENT' ? selectedChunkId : attachUrl.trim()) ? 1 : 0.6,
                    cursor: (attachTab === 'DOCUMENT' ? selectedChunkId : attachUrl.trim()) ? 'pointer' : 'not-allowed',
                  }}
                >
                  {attachTab === 'DOCUMENT' ? 'Attach Document Evidence' : 'Register Website Source'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Modal: Batch Publication Result */}
      {batchModalResult && (
        <div style={modalOverlayStyle}>
          <div style={{ ...modalBoxStyle, maxWidth: '600px' }}>
            <div style={modalHeaderStyle}>
              <h3 style={{ margin: 0, fontSize: '17px', fontWeight: 700, color: '#111827' }}>
                Batch Publication Results
              </h3>
              <button
                type="button"
                onClick={() => setBatchModalResult(null)}
                style={closeBtnStyle}
                title="Close dialog (Esc)"
              >
                ✕
              </button>
            </div>
            <div style={modalBodyStyle}>
              <div style={{ display: 'flex', gap: '12px', marginBottom: '16px', fontSize: '13px' }}>
                <span style={{ padding: '4px 8px', background: '#dcfce7', color: '#166534', borderRadius: '4px', fontWeight: 600 }}>
                  Published: {batchModalResult.published_count}
                </span>
                <span style={{ padding: '4px 8px', background: batchModalResult.failed_count > 0 ? '#fee2e2' : '#f3f4f6', color: batchModalResult.failed_count > 0 ? '#991b1b' : '#374151', borderRadius: '4px', fontWeight: 600 }}>
                  Failed: {batchModalResult.failed_count}
                </span>
                {batchModalResult.is_idempotent_retry && (
                  <span style={{ padding: '4px 8px', background: '#e0e7ff', color: '#3730a3', borderRadius: '4px', fontWeight: 600 }}>
                    Idempotent Retry Safe
                  </span>
                )}
              </div>

              <div style={{ maxHeight: '280px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '8px' }}>
                {batchModalResult.outcomes.map((item, idx) => (
                  <div
                    key={idx}
                    style={{
                      padding: '10px 12px',
                      borderRadius: '6px',
                      border: `1px solid ${item.success ? '#bbf7d0' : '#fecaca'}`,
                      background: item.success ? '#f0fdf4' : '#fef2f2',
                      fontSize: '12px',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontWeight: 600 }}>
                      <span>{item.company_name || item.note_id}</span>
                      <span style={{ color: item.success ? '#16a34a' : '#dc2626' }}>{item.status}</span>
                    </div>
                    {item.error && (
                      <div style={{ color: '#b91c1c', marginTop: '4px' }}>{item.error}</div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            <div style={modalFooterStyle}>
              <button
                type="button"
                onClick={() => setBatchModalResult(null)}
                style={btnPrimaryStyle}
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

// Helper to format safe bounded excerpt for preview
function formatSafeExcerpt(text?: string | null, maxLen: number = 130): string {
  if (!text) return ''
  const cleaned = text.replace(/\s+/g, ' ').trim()
  if (cleaned.length <= maxLen) return cleaned
  return cleaned.slice(0, maxLen).trimEnd() + '…'
}

// Styling Constants
const modalOverlayStyle: React.CSSProperties = {
  position: 'fixed',
  top: 0,
  left: 0,
  right: 0,
  bottom: 0,
  background: 'rgba(0, 0, 0, 0.5)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  zIndex: 3000,
  padding: '16px',
  backdropFilter: 'blur(2px)',
}

const modalBoxStyle: React.CSSProperties = {
  background: '#ffffff',
  borderRadius: '12px',
  width: '100%',
  maxWidth: '520px',
  maxHeight: 'min(90vh, 760px)',
  display: 'flex',
  flexDirection: 'column',
  boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.15), 0 8px 10px -6px rgba(0, 0, 0, 0.1)',
  overflow: 'hidden',
  boxSizing: 'border-box',
}

const modalHeaderStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  padding: '16px 20px',
  borderBottom: '1px solid #e5e7eb',
  background: '#ffffff',
  flexShrink: 0,
}

const modalFormStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  flex: 1,
  minHeight: 0,
  overflow: 'hidden',
}

const modalBodyStyle: React.CSSProperties = {
  padding: '18px 20px',
  overflowY: 'auto',
  overflowX: 'hidden',
  flex: 1,
}

const modalFooterStyle: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'flex-end',
  gap: '10px',
  padding: '12px 20px',
  borderTop: '1px solid #e5e7eb',
  background: '#f8fafc',
  flexShrink: 0,
}

const closeBtnStyle: React.CSSProperties = {
  background: 'none',
  border: 'none',
  fontSize: '18px',
  lineHeight: '1',
  cursor: 'pointer',
  color: '#6b7280',
  padding: '4px',
  borderRadius: '4px',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
}

const labelStyle: React.CSSProperties = {
  display: 'block',
  fontSize: '12px',
  fontWeight: 600,
  color: '#374151',
  marginBottom: '6px',
}

const inputStyle: React.CSSProperties = {
  width: '100%',
  padding: '8px 12px',
  borderRadius: '6px',
  border: '1px solid #d1d5db',
  fontSize: '13px',
  boxSizing: 'border-box',
}

const btnPrimaryStyle: React.CSSProperties = {
  background: 'var(--color-teal, #0d9488)',
  color: '#ffffff',
  border: 'none',
  padding: '8px 16px',
  borderRadius: '6px',
  fontSize: '13px',
  fontWeight: 600,
  cursor: 'pointer',
}

const btnSecondaryStyle: React.CSSProperties = {
  background: '#f3f4f6',
  color: '#374151',
  border: '1px solid #d1d5db',
  padding: '8px 16px',
  borderRadius: '6px',
  fontSize: '13px',
  fontWeight: 500,
  cursor: 'pointer',
}
