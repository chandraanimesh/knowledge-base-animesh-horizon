/**
 * Document Ingestion Panel & Multi-File Queue Review
 *
 * Implements the complete multi-step workflow:
 * Step 1: Prepare upload (Drag & drop, queue, department selection, optional metadata accordion)
 * Step 2: Review & Edit (Summary before submission if metadata is provided; fast path if blank)
 * Step 3: Confirm & Ingest (Sequential execution, duplicate submission prevention, progress reporting)
 * Step 4: Success & Completion (Auto-collapse on full success, clear state, error isolation on failures)
 */

import React, { useState, useRef, useCallback } from 'react'
import { uploadDocument, computeFileSHA256, checkDocumentHash } from '../api/documents'
import {
  QueuedFileItem,
  UploadQueueStatus,
  DocumentMetadataDraft,
  ALLOWED_DOC_TYPES,
  ALLOWED_CLASSIFICATIONS,
  ALLOWED_VISIBILITY_OPTIONS,
  formatFileSize,
  moveItemUp,
  moveItemDown,
  toggleItemSelection,
  setAllSelection,
  removeQueueItem,
  hasMetadata,
  validateMetadataDraft,
  applyBatchMetadata,
  setItemMetadata,
} from './uploadQueueDefinitions'

export type { QueuedFileItem, UploadQueueStatus, DocumentMetadataDraft }
export {
  formatFileSize,
  ALLOWED_DOC_TYPES,
  ALLOWED_CLASSIFICATIONS,
  ALLOWED_VISIBILITY_OPTIONS,
  hasMetadata,
  validateMetadataDraft,
  applyBatchMetadata,
  setItemMetadata,
}

export interface IngestionDepartmentOption {
  id: string
  code: string
  name: string
}

export interface UploadQueueReviewProps {
  queue: QueuedFileItem[]
  isProcessing: boolean
  activeFileId: string | null
  isDestinationReady?: boolean
  selectedDepartmentCode?: string
  departments?: IngestionDepartmentOption[]
  onSelectDepartment?: (deptCode: string) => void
  onAddFiles?: (files: File[]) => void
  onToggleSelect: (id: string) => void
  onSelectAll: () => void
  onDeselectAll: () => void
  onRemoveItem: (id: string) => void
  onMoveUp: (index: number) => void
  onMoveDown: (index: number) => void
  onStartIngestion: (metadataMap?: Record<string, DocumentMetadataDraft>) => void
  onRetryItem: (id: string) => void
  onClearQueue: () => void
  onClose?: () => void
  onSuccess?: (info: { filesCount: number; chunksIndexed: number }) => void
}

export function UploadQueueReview({
  queue,
  isProcessing,
  activeFileId,
  isDestinationReady = true,
  selectedDepartmentCode = '',
  departments = [],
  onSelectDepartment,
  onAddFiles,
  onToggleSelect,
  onSelectAll,
  onDeselectAll,
  onRemoveItem,
  onMoveUp,
  onMoveDown,
  onStartIngestion,
  onRetryItem,
  onClearQueue,
  onClose,
}: UploadQueueReviewProps) {
  // Step state: 'edit' (Step 1) | 'review' (Step 2)
  const [workflowStep, setWorkflowStep] = useState<'edit' | 'review'>('edit')

  // Metadata form state
  const [isMetadataExpanded, setIsMetadataExpanded] = useState(true)
  const [batchScope, setBatchScope] = useState<'all' | 'single'>('all')
  const [targetFileId, setTargetFileId] = useState<string>(queue[0]?.id || '')

  // Draft metadata values
  const [docType, setDocType] = useState<string>('Reference Data')
  const [classification, setClassification] = useState<string>('Internal')
  const [visibility, setVisibility] = useState<string>('Department only')
  const [tagInput, setTagInput] = useState<string>('')
  const [tags, setTags] = useState<string[]>([])
  const [description, setDescription] = useState<string>('')

  // Drag and drop state
  const [isDragOver, setIsDragOver] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Current draft object
  const currentDraft: DocumentMetadataDraft = {
    docType: docType || undefined,
    classification: classification || undefined,
    tags,
    visibility: visibility || undefined,
    description: description.trim() || undefined,
    isSuggested: false,
  }

  const isMetadataPopulated = hasMetadata(currentDraft)
  const { warnings } = validateMetadataDraft(currentDraft)

  const selectedCount = queue.filter((item) => item.selected).length
  const completedCount = queue.filter((item) => item.status === 'completed').length
  const failedCount = queue.filter((item) => item.status === 'failed').length
  const unprocessedCount = queue.filter(
    (item) => item.status === 'pending' || (item.status === 'failed' && !item.selected)
  ).length

  const eligibleToProcess = queue.filter(
    (item) => item.selected && item.status !== 'completed'
  ).length

  // Tag handler
  const handleAddTag = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      const clean = tagInput.trim().toLowerCase().replace(/[^a-z0-9_-]/g, '')
      if (clean && !tags.includes(clean) && tags.length < 20) {
        setTags([...tags, clean])
        setTagInput('')
      }
    }
  }

  const handleRemoveTag = (tagToRemove: string) => {
    setTags(tags.filter((t) => t !== tagToRemove))
  }

  // Drag & drop handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(false)
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onAddFiles?.(Array.from(e.dataTransfer.files))
    }
  }

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onAddFiles?.(Array.from(e.target.files))
      e.target.value = ''
    }
  }

  // Submission handler
  const handleProceedOrReview = () => {
    if (isMetadataPopulated) {
      setWorkflowStep('review')
    } else {
      // Fast path: Ingest directly if all metadata is blank
      handleConfirmSubmission()
    }
  }

  const handleConfirmSubmission = () => {
    const metaMap: Record<string, DocumentMetadataDraft> = {}
    if (isMetadataPopulated) {
      if (batchScope === 'all') {
        queue.forEach((q) => {
          metaMap[q.id] = { ...currentDraft, tags: [...tags] }
        })
      } else if (targetFileId) {
        metaMap[targetFileId] = { ...currentDraft, tags: [...tags] }
      }
    }
    onStartIngestion(metaMap)
  }

  return (
    <div
      role="region"
      aria-label="Upload documents to RAG"
      style={{
        background: '#FFFFFF',
        border: '1px solid #E5E7EB',
        borderRadius: '12px',
        padding: '16px 20px',
        marginBottom: '10px',
        boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.08), 0 8px 10px -6px rgba(0, 0, 0, 0.04)',
        fontFamily: "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        boxSizing: 'border-box',
        display: 'flex',
        flexDirection: 'column',
        gap: '14px',
        width: '100%',
        maxWidth: '820px',
        margin: '0 auto 10px auto',
      }}
    >
      {/* Hidden File Input */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        style={{ display: 'none' }}
        onChange={handleFileInputChange}
      />

      {/* 1. Header Toolbar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          borderBottom: '1px solid #F1F5F9',
          paddingBottom: '10px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div
            style={{
              width: '32px',
              height: '32px',
              borderRadius: '8px',
              background: '#EFF6FF',
              color: '#2563EB',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexShrink: 0,
            }}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
          </div>
          <div>
            <h3 style={{ margin: 0, fontSize: '14px', fontWeight: 600, color: '#111827' }}>
              {workflowStep === 'review' ? 'Review & Confirm Ingestion' : 'Upload documents to RAG'}
            </h3>
            <p style={{ margin: '1px 0 0 0', fontSize: '11px', color: '#64748B' }}>
              {workflowStep === 'review'
                ? 'Verify document metadata and target department before final indexing.'
                : 'Files will be parsed, chunked and indexed for search and chat.'}
            </p>
          </div>
        </div>

        {onClose && !isProcessing && (
          <button
            type="button"
            onClick={onClose}
            aria-label="Close upload panel"
            style={{
              background: 'transparent',
              border: 'none',
              cursor: 'pointer',
              color: '#9CA3AF',
              fontSize: '18px',
              padding: '4px',
              lineHeight: 1,
              borderRadius: '4px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
            onMouseEnter={(e) => (e.currentTarget.style.color = '#374151')}
            onMouseLeave={(e) => (e.currentTarget.style.color = '#9CA3AF')}
          >
            ✕
          </button>
        )}
      </div>

      {/* ========================================================================= */}
      {/* STEP 1: PREPARATION (Two-Column Layout)                                   */}
      {/* ========================================================================= */}
      {workflowStep === 'edit' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', alignItems: 'start' }}>
          {/* Left Column: Upload Department, Dropzone & Queue */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {/* Department Select */}
            <div>
              <label
                htmlFor="rag-upload-dept-select"
                style={{ display: 'block', fontSize: '11px', fontWeight: 600, color: '#374151', marginBottom: '4px' }}
              >
                Upload to Department
              </label>
              <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
                <span
                  style={{
                    position: 'absolute',
                    left: '8px',
                    color: '#6B7280',
                    display: 'flex',
                    alignItems: 'center',
                    pointerEvents: 'none',
                  }}
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="4" y="2" width="16" height="20" rx="2" ry="2" />
                    <line x1="9" y1="22" x2="9" y2="12" />
                    <line x1="15" y1="22" x2="15" y2="12" />
                    <line x1="9" y1="6" x2="9" y2="6" />
                    <line x1="15" y1="6" x2="15" y2="6" />
                    <line x1="9" y1="10" x2="9" y2="10" />
                    <line x1="15" y1="10" x2="15" y2="10" />
                  </svg>
                </span>
                <select
                  id="rag-upload-dept-select"
                  value={selectedDepartmentCode}
                  onChange={(e) => onSelectDepartment?.(e.target.value)}
                  disabled={isProcessing}
                  style={{
                    width: '100%',
                    padding: '6px 10px 6px 26px',
                    borderRadius: '6px',
                    border: '1px solid #D1D5DB',
                    background: '#FFFFFF',
                    fontSize: '11.5px',
                    color: '#374151',
                    outline: 'none',
                    cursor: isProcessing ? 'default' : 'pointer',
                  }}
                >
                  <option value="">Select Department</option>
                  {departments.map((dept) => (
                    <option key={dept.id} value={dept.code}>
                      {dept.name} ({dept.code})
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Drag & Drop Zone */}
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => {
                if (!isProcessing) fileInputRef.current?.click()
              }}
              style={{
                border: isDragOver ? '1.5px dashed #2563EB' : '1.5px dashed #CBD5E1',
                borderRadius: '8px',
                background: isDragOver ? '#EFF6FF' : '#F8FAFC',
                padding: '14px 12px',
                textAlign: 'center',
                cursor: isProcessing ? 'default' : 'pointer',
                transition: 'all 0.15s ease',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '4px',
              }}
            >
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#2563EB" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
              <div style={{ fontSize: '11.5px', fontWeight: 500, color: '#1F2937' }}>
                Drag & drop files here or <span style={{ color: '#2563EB', textDecoration: 'underline' }}>click to browse</span>
              </div>
              <div style={{ fontSize: '9.5px', color: '#9CA3AF' }}>
                PDF, DOCX, TXT, MD, CSV, XLSX, XLS (Max 100 MB per file)
              </div>
            </div>

            {/* Upload Queue Section */}
            <div
              style={{
                border: '1px solid #E5E7EB',
                borderRadius: '8px',
                padding: '8px 10px',
                background: '#FFFFFF',
                display: 'flex',
                flexDirection: 'column',
                gap: '6px',
              }}
            >
              {/* Queue Header */}
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  paddingBottom: '5px',
                  borderBottom: '1px solid #F1F5F9',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span style={{ fontSize: '11.5px', fontWeight: 600, color: '#111827' }}>Upload Queue</span>
                  <span
                    style={{
                      fontSize: '10px',
                      color: '#4B5563',
                      background: '#F3F4F6',
                      padding: '1px 6px',
                      borderRadius: '10px',
                      fontWeight: 500,
                    }}
                  >
                    {selectedCount} of {queue.length} selected
                  </span>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <button
                    type="button"
                    onClick={onSelectAll}
                    disabled={isProcessing}
                    style={{
                      background: 'transparent',
                      border: 'none',
                      color: isProcessing ? '#D1D5DB' : '#2563EB',
                      fontSize: '10.5px',
                      cursor: isProcessing ? 'default' : 'pointer',
                      padding: '2px 4px',
                      fontWeight: 500,
                    }}
                  >
                    Select All
                  </button>
                  <span style={{ color: '#E5E7EB', fontSize: '10px' }}>|</span>
                  <button
                    type="button"
                    onClick={onDeselectAll}
                    disabled={isProcessing}
                    style={{
                      background: 'transparent',
                      border: 'none',
                      color: isProcessing ? '#D1D5DB' : '#6B7280',
                      fontSize: '10.5px',
                      cursor: isProcessing ? 'default' : 'pointer',
                      padding: '2px 4px',
                      fontWeight: 500,
                    }}
                  >
                    Deselect All
                  </button>
                  {!isProcessing && queue.length > 0 && (
                    <>
                      <span style={{ color: '#E5E7EB', fontSize: '10px' }}>|</span>
                      <button
                        type="button"
                        onClick={onClearQueue}
                        title="Clear all files from queue"
                        style={{
                          background: 'transparent',
                          border: 'none',
                          color: '#9CA3AF',
                          fontSize: '10.5px',
                          cursor: 'pointer',
                          padding: '2px 4px',
                        }}
                      >
                        Clear
                      </button>
                    </>
                  )}
                </div>
              </div>

              {/* Scrollable File List */}
              <div
                style={{
                  maxHeight: '140px',
                  overflowY: 'auto',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '4px',
                }}
              >
                {queue.length === 0 ? (
                  <div style={{ fontSize: '10.5px', color: '#9CA3AF', textAlign: 'center', padding: '12px' }}>
                    No files queued yet. Select or drop files above.
                  </div>
                ) : (
                  queue.map((item, index) => {
                    const isActive = item.id === activeFileId
                    const isCompleted = item.status === 'completed'
                    const isFailed = item.status === 'failed'

                    return (
                      <div
                        key={item.id}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '6px',
                          padding: '4px 6px',
                          borderRadius: '6px',
                          background: isActive ? '#EFF6FF' : isCompleted ? '#F0FDF4' : '#F9FAFB',
                          border: isActive
                            ? '1px solid #BFDBFE'
                            : isCompleted
                            ? '1px solid #BBF7D0'
                            : isFailed
                            ? '1px solid #FECACA'
                            : '1px solid #F3F4F6',
                        }}
                      >
                        <input
                          type="checkbox"
                          checked={item.selected}
                          disabled={isProcessing || isCompleted}
                          onChange={() => onToggleSelect(item.id)}
                          aria-label={`Select ${item.name}`}
                          style={{
                            cursor: isProcessing || isCompleted ? 'default' : 'pointer',
                            accentColor: '#2563EB',
                            margin: 0,
                            width: '12px',
                            height: '12px',
                          }}
                        />

                        {/* File icon */}
                        <svg
                          width="12"
                          height="12"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke={isCompleted ? '#16A34A' : isActive ? '#2563EB' : '#64748B'}
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        >
                          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                          <polyline points="14 2 14 8 20 8" />
                        </svg>

                        <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
                          <div style={{ display: 'flex', alignItems: 'baseline', gap: '4px' }}>
                            <span
                              title={item.name}
                              style={{
                                fontSize: '10.5px',
                                fontWeight: 600,
                                color: isFailed ? '#991B1B' : '#1F2937',
                                whiteSpace: 'nowrap',
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                              }}
                            >
                              {item.name}
                            </span>
                            <span style={{ fontSize: '9px', color: '#9CA3AF' }}>{formatFileSize(item.size)}</span>
                          </div>
                          <div style={{ fontSize: '9px', marginTop: '1px' }}>
                            {item.status === 'uploading' && (
                              <span style={{ color: '#2563EB', fontWeight: 500 }}>Processing…</span>
                            )}
                            {item.status === 'completed' && (
                              <span style={{ color: '#16A34A', fontWeight: 500 }}>
                                ✓ {item.skipped ? 'Already indexed' : `${item.chunksCreated ?? 0} chunks indexed`}
                              </span>
                            )}
                            {item.status === 'failed' && (
                              <span style={{ color: '#DC2626', fontWeight: 500 }} title={item.errorMessage}>
                                ✕ {item.errorMessage || 'Upload failed'}
                              </span>
                            )}
                            {item.status === 'pending' && (
                              <span style={{ color: item.selected ? '#6B7280' : '#9CA3AF' }}>
                                {item.selected ? 'Queued' : 'Unchecked'}
                              </span>
                            )}
                          </div>
                        </div>

                        {/* Row Actions */}
                        <div style={{ display: 'flex', alignItems: 'center', gap: '2px' }}>
                          {isFailed && !isProcessing && (
                            <button
                              type="button"
                              onClick={() => onRetryItem(item.id)}
                              style={{
                                background: '#FEF2F2',
                                border: '1px solid #FECACA',
                                color: '#DC2626',
                                borderRadius: '3px',
                                fontSize: '9px',
                                padding: '1px 4px',
                                cursor: 'pointer',
                              }}
                            >
                              Retry
                            </button>
                          )}
                          {!isProcessing && !isCompleted && (
                            <>
                              <button
                                type="button"
                                onClick={() => onMoveUp(index)}
                                disabled={index === 0}
                                style={{ background: 'transparent', border: 'none', color: '#9CA3AF', fontSize: '9px', cursor: 'pointer', padding: '1px' }}
                              >
                                ▲
                              </button>
                              <button
                                type="button"
                                onClick={() => onMoveDown(index)}
                                disabled={index === queue.length - 1}
                                style={{ background: 'transparent', border: 'none', color: '#9CA3AF', fontSize: '9px', cursor: 'pointer', padding: '1px' }}
                              >
                                ▼
                              </button>
                              <button
                                type="button"
                                onClick={() => onRemoveItem(item.id)}
                                style={{ background: 'transparent', border: 'none', color: '#9CA3AF', fontSize: '10px', cursor: 'pointer', padding: '1px' }}
                              >
                                ✕
                              </button>
                            </>
                          )}
                        </div>
                      </div>
                    )
                  })
                )}
              </div>

              {/* Queue Footer */}
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  borderTop: '1px solid #F1F5F9',
                  paddingTop: '6px',
                  marginTop: '2px',
                }}
              >
                <div style={{ fontSize: '9.5px', color: '#6B7280', display: 'flex', gap: '6px' }}>
                  {completedCount > 0 && <span style={{ color: '#16A34A', fontWeight: 600 }}>{completedCount} completed</span>}
                  {failedCount > 0 && <span style={{ color: '#DC2626', fontWeight: 600 }}>{failedCount} failed</span>}
                  {unprocessedCount > 0 && <span>{unprocessedCount} queued</span>}
                </div>

                <button
                  type="button"
                  onClick={handleProceedOrReview}
                  disabled={isProcessing || eligibleToProcess === 0 || !isDestinationReady}
                  style={{
                    background: isProcessing || eligibleToProcess === 0 || !isDestinationReady ? '#F3F4F6' : '#2563EB',
                    color: isProcessing || eligibleToProcess === 0 || !isDestinationReady ? '#9CA3AF' : '#FFFFFF',
                    border: 'none',
                    borderRadius: '6px',
                    fontSize: '11px',
                    fontWeight: 600,
                    padding: '5px 12px',
                    cursor: isProcessing || eligibleToProcess === 0 || !isDestinationReady ? 'default' : 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px',
                  }}
                >
                  {isProcessing ? (
                    'Ingesting…'
                  ) : !isDestinationReady ? (
                    'Select Department'
                  ) : isMetadataPopulated ? (
                    'Review & Ingest'
                  ) : (
                    `Ingest Files`
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Right Column: Document Details & Metadata (optional) */}
          <div
            style={{
              border: '1px solid #E5E7EB',
              borderRadius: '8px',
              padding: '10px 12px',
              background: '#FFFFFF',
              display: 'flex',
              flexDirection: 'column',
              gap: '10px',
            }}
          >
            {/* Collapsible Accordion Header */}
            <div
              onClick={() => setIsMetadataExpanded(!isMetadataExpanded)}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                cursor: 'pointer',
                userSelect: 'none',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <span style={{ fontSize: '11px', color: '#2563EB' }}>{isMetadataExpanded ? '▾' : '▸'}</span>
                <span style={{ fontSize: '11.5px', fontWeight: 600, color: '#111827' }}>
                  Document Details & Metadata (optional)
                </span>
              </div>
              <span style={{ fontSize: '10px', color: '#6B7280' }}>
                {isMetadataPopulated ? '● Metadata entered' : 'Optional'}
              </span>
            </div>

            {isMetadataExpanded && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {/* Batch Scope Selector (if multiple files) */}
                {queue.length > 1 && (
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      background: '#F8FAFC',
                      padding: '4px 8px',
                      borderRadius: '6px',
                      border: '1px solid #E2E8F0',
                      fontSize: '10.5px',
                    }}
                  >
                    <span style={{ color: '#475569', fontWeight: 500 }}>Apply metadata to:</span>
                    <div style={{ display: 'flex', gap: '6px' }}>
                      <label style={{ display: 'flex', alignItems: 'center', gap: '3px', cursor: 'pointer' }}>
                        <input
                          type="radio"
                          name="batchScope"
                          checked={batchScope === 'all'}
                          onChange={() => setBatchScope('all')}
                        />
                        All selected files
                      </label>
                      <label style={{ display: 'flex', alignItems: 'center', gap: '3px', cursor: 'pointer' }}>
                        <input
                          type="radio"
                          name="batchScope"
                          checked={batchScope === 'single'}
                          onChange={() => {
                            setBatchScope('single')
                            if (!targetFileId && queue[0]) setTargetFileId(queue[0].id)
                          }}
                        />
                        Single file
                      </label>
                    </div>
                  </div>
                )}

                {batchScope === 'single' && queue.length > 1 && (
                  <div>
                    <label style={{ display: 'block', fontSize: '10px', color: '#6B7280', marginBottom: '2px' }}>
                      Select target file:
                    </label>
                    <select
                      value={targetFileId}
                      onChange={(e) => setTargetFileId(e.target.value)}
                      style={{
                        width: '100%',
                        padding: '4px 6px',
                        borderRadius: '5px',
                        border: '1px solid #D1D5DB',
                        fontSize: '11px',
                      }}
                    >
                      {queue.map((q) => (
                        <option key={q.id} value={q.id}>
                          {q.name}
                        </option>
                      ))}
                    </select>
                  </div>
                )}

                {/* Form Grid: Doc Type & Classification */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                  <div>
                    <label style={{ display: 'block', fontSize: '10.5px', fontWeight: 600, color: '#374151', marginBottom: '2px' }}>
                      Document Type <span title="Descriptive document classification for organizational retrieval" style={{ color: '#9CA3AF', cursor: 'help' }}>ⓘ</span>
                    </label>
                    <select
                      value={docType}
                      onChange={(e) => setDocType(e.target.value)}
                      style={{
                        width: '100%',
                        padding: '5px 8px',
                        borderRadius: '6px',
                        border: '1px solid #D1D5DB',
                        fontSize: '11px',
                        color: '#374151',
                      }}
                    >
                      {ALLOWED_DOC_TYPES.map((t) => (
                        <option key={t} value={t}>
                          {t}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label style={{ display: 'block', fontSize: '10.5px', fontWeight: 600, color: '#374151', marginBottom: '2px' }}>
                      Classification <span title="Informational sensitivity label. Does not bypass department permissions." style={{ color: '#9CA3AF', cursor: 'help' }}>ⓘ</span>
                    </label>
                    <select
                      value={classification}
                      onChange={(e) => setClassification(e.target.value)}
                      style={{
                        width: '100%',
                        padding: '5px 8px',
                        borderRadius: '6px',
                        border: '1px solid #D1D5DB',
                        fontSize: '11px',
                        color: '#374151',
                      }}
                    >
                      {ALLOWED_CLASSIFICATIONS.map((c) => (
                        <option key={c} value={c}>
                          {c}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                {/* Tags & Visibility Intent */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                  <div>
                    <label style={{ display: 'block', fontSize: '10.5px', fontWeight: 600, color: '#374151', marginBottom: '2px' }}>
                      Tags <span title="Enter keywords and press Enter to add" style={{ color: '#9CA3AF', cursor: 'help' }}>ⓘ</span>
                    </label>
                    <input
                      type="text"
                      placeholder="Add tags (press Enter)"
                      value={tagInput}
                      onChange={(e) => setTagInput(e.target.value)}
                      onKeyDown={handleAddTag}
                      style={{
                        width: '100%',
                        padding: '5px 8px',
                        borderRadius: '6px',
                        border: '1px solid #D1D5DB',
                        fontSize: '11px',
                        boxSizing: 'border-box',
                      }}
                    />
                    {tags.length > 0 && (
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginTop: '4px' }}>
                        {tags.map((t) => (
                          <span
                            key={t}
                            style={{
                              fontSize: '10px',
                              background: '#EFF6FF',
                              color: '#1D4ED8',
                              border: '1px solid #BFDBFE',
                              borderRadius: '4px',
                              padding: '1px 5px',
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: '3px',
                            }}
                          >
                            {t}
                            <span
                              onClick={() => handleRemoveTag(t)}
                              style={{ cursor: 'pointer', fontWeight: 700 }}
                            >
                              ✕
                            </span>
                          </span>
                        ))}
                      </div>
                    )}
                  </div>

                  <div>
                    <label style={{ display: 'block', fontSize: '10.5px', fontWeight: 600, color: '#374151', marginBottom: '2px' }}>
                      Visibility Intent <span title="Target scope within organizational hierarchy" style={{ color: '#9CA3AF', cursor: 'help' }}>ⓘ</span>
                    </label>
                    <select
                      value={visibility}
                      onChange={(e) => setVisibility(e.target.value)}
                      style={{
                        width: '100%',
                        padding: '5px 8px',
                        borderRadius: '6px',
                        border: '1px solid #D1D5DB',
                        fontSize: '11px',
                        color: '#374151',
                      }}
                    >
                      {ALLOWED_VISIBILITY_OPTIONS.map((v) => (
                        <option key={v} value={v}>
                          {v}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                {/* Description / Context */}
                <div>
                  <label style={{ display: 'block', fontSize: '10.5px', fontWeight: 600, color: '#374151', marginBottom: '2px' }}>
                    Description / Context <span title="Optional descriptive summary for search enrichment" style={{ color: '#9CA3AF', cursor: 'help' }}>ⓘ</span>
                  </label>
                  <textarea
                    rows={2}
                    placeholder="Enter document context, notes, or purpose…"
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '5px 8px',
                      borderRadius: '6px',
                      border: '1px solid #D1D5DB',
                      fontSize: '11px',
                      boxSizing: 'border-box',
                      resize: 'none',
                    }}
                  />
                </div>

                {/* Info Callout */}
                <div
                  style={{
                    background: '#EFF6FF',
                    border: '1px solid #DBEAFE',
                    borderRadius: '6px',
                    padding: '6px 8px',
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: '6px',
                    marginTop: '2px',
                  }}
                >
                  <span style={{ color: '#2563EB', fontSize: '12px' }}>ⓘ</span>
                  <div style={{ fontSize: '10px', color: '#1E40AF', lineHeight: 1.4 }}>
                    Metadata helps with organization and retrieval. It does not change department access or permissions.
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* STEP 2: REVIEW & CONFIRM SCREEN                                          */}
      {/* ========================================================================= */}
      {workflowStep === 'review' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <div
            style={{
              background: '#F8FAFC',
              border: '1px solid #E2E8F0',
              borderRadius: '8px',
              padding: '12px 14px',
              display: 'flex',
              flexDirection: 'column',
              gap: '10px',
            }}
          >
            {/* Summary Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
              <div>
                <span style={{ fontSize: '10px', color: '#64748B', fontWeight: 600 }}>TARGET DEPARTMENT</span>
                <div style={{ fontSize: '12px', fontWeight: 600, color: '#1E293B', marginTop: '2px' }}>
                  {departments.find((d) => d.code === selectedDepartmentCode)?.name || selectedDepartmentCode || 'None Selected'}{' '}
                  <span style={{ fontSize: '10px', color: '#64748B', fontFamily: 'monospace' }}>({selectedDepartmentCode})</span>
                </div>
              </div>

              <div>
                <span style={{ fontSize: '10px', color: '#64748B', fontWeight: 600 }}>FILES TO INGEST</span>
                <div style={{ fontSize: '12px', fontWeight: 600, color: '#1E293B', marginTop: '2px' }}>
                  {queue.filter((q) => q.selected).length} selected file(s)
                </div>
              </div>
            </div>

            <div style={{ borderTop: '1px solid #E2E8F0', paddingTop: '8px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
              <div>
                <span style={{ fontSize: '10px', color: '#64748B', fontWeight: 600 }}>DOCUMENT TYPE</span>
                <div style={{ fontSize: '11.5px', color: '#334155', marginTop: '2px' }}>
                  {currentDraft.docType || 'Not specified'}
                </div>
              </div>

              <div>
                <span style={{ fontSize: '10px', color: '#64748B', fontWeight: 600 }}>CLASSIFICATION & VISIBILITY</span>
                <div style={{ display: 'flex', gap: '4px', marginTop: '2px' }}>
                  <span style={{ fontSize: '10px', padding: '1px 6px', background: '#F1F5F9', borderRadius: '4px', color: '#475569' }}>
                    {currentDraft.classification || 'Internal'}
                  </span>
                  <span style={{ fontSize: '10px', padding: '1px 6px', background: '#EFF6FF', borderRadius: '4px', color: '#1D4ED8' }}>
                    {currentDraft.visibility || 'Department only'}
                  </span>
                </div>
              </div>
            </div>

            {/* Tags */}
            <div style={{ borderTop: '1px solid #E2E8F0', paddingTop: '8px' }}>
              <span style={{ fontSize: '10px', color: '#64748B', fontWeight: 600 }}>TAGS</span>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginTop: '3px' }}>
                {tags.length > 0 ? (
                  tags.map((t) => (
                    <span
                      key={t}
                      style={{
                        fontSize: '10px',
                        background: '#EFF6FF',
                        color: '#1D4ED8',
                        padding: '1px 6px',
                        borderRadius: '4px',
                        border: '1px solid #BFDBFE',
                      }}
                    >
                      {t}
                    </span>
                  ))
                ) : (
                  <span style={{ fontSize: '11px', color: '#9CA3AF' }}>No tags specified</span>
                )}
              </div>
            </div>

            {/* Description */}
            <div style={{ borderTop: '1px solid #E2E8F0', paddingTop: '8px' }}>
              <span style={{ fontSize: '10px', color: '#64748B', fontWeight: 600 }}>DESCRIPTION / CONTEXT</span>
              <div style={{ fontSize: '11.5px', color: '#334155', marginTop: '2px', whiteSpace: 'pre-wrap' }}>
                {currentDraft.description || <span style={{ color: '#9CA3AF' }}>No description provided</span>}
              </div>
            </div>

            {/* Warnings if any */}
            {warnings.length > 0 && (
              <div style={{ background: '#FFFBEB', border: '1px solid #FCD34D', borderRadius: '6px', padding: '6px 8px' }}>
                {warnings.map((w, idx) => (
                  <div key={idx} style={{ fontSize: '10px', color: '#B45309' }}>
                    ⚠️ {w}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Review Actions */}
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px', marginTop: '4px' }}>
            <button
              type="button"
              onClick={() => setWorkflowStep('edit')}
              disabled={isProcessing}
              style={{
                padding: '6px 14px',
                borderRadius: '6px',
                background: '#FFFFFF',
                border: '1px solid #D1D5DB',
                color: '#374151',
                fontSize: '11.5px',
                fontWeight: 500,
                cursor: 'pointer',
              }}
            >
              Back to edit
            </button>
            <button
              type="button"
              onClick={handleConfirmSubmission}
              disabled={isProcessing || !isDestinationReady}
              style={{
                padding: '6px 16px',
                borderRadius: '6px',
                background: isProcessing ? '#94A3B8' : '#2563EB',
                border: 'none',
                color: '#FFFFFF',
                fontSize: '11.5px',
                fontWeight: 600,
                cursor: isProcessing ? 'default' : 'pointer',
              }}
            >
              {isProcessing ? 'Ingesting…' : 'Confirm & ingest'}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

/**
 * State hook for managing multi-file upload queue, ordering, selection, and sequential execution.
 */
export function useUploadQueue() {
  const [queue, setQueue] = useState<QueuedFileItem[]>([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [activeFileId, setActiveFileId] = useState<string | null>(null)
  const isProcessingRef = useRef(false)

  const addFiles = useCallback((files: File[]) => {
    const newItems: QueuedFileItem[] = files.map((file) => ({
      id: `q-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      file,
      name: file.name,
      size: file.size,
      selected: true,
      status: 'pending',
    }))
    setQueue((prev) => [...prev, ...newItems])
  }, [])

  const toggleSelect = useCallback((id: string) => {
    if (isProcessingRef.current) return
    setQueue((prev) =>
      prev.map((item) => (item.id === id ? { ...item, selected: !item.selected } : item))
    )
  }, [])

  const selectAll = useCallback(() => {
    if (isProcessingRef.current) return
    setQueue((prev) => prev.map((item) => ({ ...item, selected: true })))
  }, [])

  const deselectAll = useCallback(() => {
    if (isProcessingRef.current) return
    setQueue((prev) => prev.map((item) => ({ ...item, selected: false })))
  }, [])

  const removeItem = useCallback((id: string) => {
    if (isProcessingRef.current) return
    setQueue((prev) => prev.filter((item) => item.id !== id))
  }, [])

  const moveUp = useCallback((index: number) => {
    if (isProcessingRef.current || index <= 0) return
    setQueue((prev) => {
      const copy = [...prev]
      const temp = copy[index - 1]
      copy[index - 1] = copy[index]
      copy[index] = temp
      return copy
    })
  }, [])

  const moveDown = useCallback((index: number) => {
    setQueue((prev) => {
      if (isProcessingRef.current || index >= prev.length - 1) return prev
      const copy = [...prev]
      const temp = copy[index + 1]
      copy[index + 1] = copy[index]
      copy[index] = temp
      return copy
    })
  }, [])

  const clearQueue = useCallback(() => {
    if (isProcessingRef.current) return
    setQueue([])
  }, [])

  const startIngestion = useCallback(
    async ({
      departmentCode,
      token,
      workspaceId,
      metadataMap,
      onCompleted,
      onSuccess,
    }: {
      departmentCode: string
      token: string
      workspaceId?: string
      metadataMap?: Record<string, DocumentMetadataDraft>
      onCompleted?: (result: {
        totalProcessed: number
        successCount: number
        failureCount: number
        totalChunks: number
      }) => void
      onSuccess?: (info: { filesCount: number; chunksIndexed: number }) => void
    }) => {
      // Prevent duplicate submissions and concurrent runners
      if (isProcessingRef.current) return
      isProcessingRef.current = true
      setIsProcessing(true)

      let totalProcessed = 0
      let successCount = 0
      let failureCount = 0
      let totalChunks = 0

      try {
        let queueSnapshot = [...queue]

        for (let i = 0; i < queueSnapshot.length; i++) {
          const item = queueSnapshot[i]
          // Process only checked files that are not already completed
          if (!item.selected || item.status === 'completed') {
            continue
          }

          totalProcessed++
          setActiveFileId(item.id)
          setQueue((prev) =>
            prev.map((q) =>
              q.id === item.id ? { ...q, status: 'uploading', errorMessage: undefined } : q
            )
          )

          // Resolve metadata payload
          const itemMeta = metadataMap?.[item.id] || item.metadata

          try {
            // Client-side SHA-256 pre-flight deduplication check
            try {
              const fileHash = await computeFileSHA256(item.file)
              const hashCheck = await checkDocumentHash(fileHash, departmentCode, token, workspaceId)
              if (hashCheck.exists) {
                successCount++
                setQueue((prev) =>
                  prev.map((q) =>
                    q.id === item.id
                      ? {
                          ...q,
                          status: 'completed',
                          chunksCreated: 0,
                          skipped: true,
                          errorMessage: `Pre-check: Already indexed in ${hashCheck.scope || 'department'}. Upload bypassed.`,
                        }
                      : q
                  )
                )
                continue
              }
            } catch (hashErr) {
              // Non-blocking: If hash pre-check fails (e.g. offline/network), proceed to normal upload
              console.warn('Hash pre-check failed, continuing with normal upload:', hashErr)
            }

            const resp = await uploadDocument(
              item.file,
              departmentCode,
              token,
              workspaceId,
              itemMeta
                ? {
                    doc_type: itemMeta.docType,
                    tags: itemMeta.tags,
                    description: itemMeta.description,
                    classification: itemMeta.classification,
                    visibility: itemMeta.visibility,
                  }
                : undefined
            )

            successCount++
            const created = resp.chunks_created || 0
            totalChunks += created

            setQueue((prev) =>
              prev.map((q) =>
                q.id === item.id
                  ? {
                      ...q,
                      status: 'completed',
                      chunksCreated: created,
                      skipped: resp.skipped,
                    }
                  : q
              )
            )
          } catch (err: any) {
            failureCount++
            const msg = err.message || 'Upload failed'
            setQueue((prev) =>
              prev.map((q) =>
                q.id === item.id
                  ? {
                      ...q,
                      status: 'failed',
                      errorMessage: msg,
                    }
                  : q
              )
            )
          }
        }

        const summary = {
          totalProcessed,
          successCount,
          failureCount,
          totalChunks,
        }

        onCompleted?.(summary)

        // Close panel and notify only if there are no failures and at least 1 success
        if (failureCount === 0 && successCount > 0) {
          onSuccess?.({
            filesCount: successCount,
            chunksIndexed: totalChunks,
          })
        }
      } finally {
        isProcessingRef.current = false
        setIsProcessing(false)
        setActiveFileId(null)
      }
    },
    [queue]
  )

  const retryItem = useCallback(
    async (
      id: string,
      {
        departmentCode,
        token,
        workspaceId,
        onCompleted,
        onSuccess,
      }: {
        departmentCode: string
        token: string
        workspaceId?: string
        onCompleted?: () => void
        onSuccess?: (info: { filesCount: number; chunksIndexed: number }) => void
      }
    ) => {
      if (isProcessingRef.current) return
      const target = queue.find((q) => q.id === id)
      if (!target) return

      isProcessingRef.current = true
      setIsProcessing(true)
      setActiveFileId(id)

      setQueue((prev) =>
        prev.map((q) =>
          q.id === id ? { ...q, status: 'uploading', errorMessage: undefined } : q
        )
      )

      try {
        try {
          const fileHash = await computeFileSHA256(target.file)
          const hashCheck = await checkDocumentHash(fileHash, departmentCode, token, workspaceId)
          if (hashCheck.exists) {
            setQueue((prev) =>
              prev.map((q) =>
                q.id === id
                  ? {
                      ...q,
                      status: 'completed',
                      chunksCreated: 0,
                      skipped: true,
                      errorMessage: `Pre-check: Already indexed in ${hashCheck.scope || 'department'}. Upload bypassed.`,
                    }
                  : q
              )
            )
            onCompleted?.()
            return
          }
        } catch (hashErr) {
          console.warn('Hash pre-check failed, continuing with normal upload:', hashErr)
        }

        const resp = await uploadDocument(
          target.file,
          departmentCode,
          token,
          workspaceId,
          target.metadata
            ? {
                doc_type: target.metadata.docType,
                tags: target.metadata.tags,
                description: target.metadata.description,
                classification: target.metadata.classification,
                visibility: target.metadata.visibility,
              }
            : undefined
        )

        const created = resp.chunks_created || 0

        setQueue((prev) =>
          prev.map((q) =>
            q.id === id
              ? {
                  ...q,
                  status: 'completed',
                  chunksCreated: created,
                  skipped: resp.skipped,
                }
              : q
          )
        )
        onCompleted?.()
      } catch (err: any) {
        setQueue((prev) =>
          prev.map((q) =>
            q.id === id
              ? { ...q, status: 'failed', errorMessage: err.message || 'Retry failed' }
              : q
          )
        )
      } finally {
        isProcessingRef.current = false
        setIsProcessing(false)
        setActiveFileId(null)
      }
    },
    [queue]
  )

  return {
    queue,
    setQueue,
    isProcessing,
    activeFileId,
    addFiles,
    toggleSelect,
    selectAll,
    deselectAll,
    removeItem,
    moveUp,
    moveDown,
    clearQueue,
    startIngestion,
    retryItem,
  }
}
