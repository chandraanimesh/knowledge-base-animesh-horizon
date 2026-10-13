/**
 * Manual Queue Selection and Ordering for Multi-File Ingestion
 *
 * Provides explicit user control before, during, and after document ingestion:
 * - Multi-file review with checkboxes (Select All / Deselect All)
 * - Move Up / Move Down accessible reordering controls
 * - Remove and Retry individual file actions
 * - Real-time sequential execution with failure isolation
 * - Subtly faded pending items, active spinner, and detailed status counts
 */

import React, { useState, useRef, useCallback } from 'react'
import { uploadDocument } from '../api/documents'
import {
  QueuedFileItem,
  UploadQueueStatus,
  formatFileSize,
  moveItemUp,
  moveItemDown,
  toggleItemSelection,
  setAllSelection,
  removeQueueItem,
} from './uploadQueueDefinitions'

export type { QueuedFileItem, UploadQueueStatus }
export { formatFileSize }

interface UploadQueueReviewProps {
  queue: QueuedFileItem[]
  isProcessing: boolean
  activeFileId: string | null
  isDestinationReady?: boolean
  onToggleSelect: (id: string) => void
  onSelectAll: () => void
  onDeselectAll: () => void
  onRemoveItem: (id: string) => void
  onMoveUp: (index: number) => void
  onMoveDown: (index: number) => void
  onStartIngestion: () => void
  onRetryItem: (id: string) => void
  onClearQueue: () => void
}

export function UploadQueueReview({
  queue,
  isProcessing,
  activeFileId,
  isDestinationReady = true,
  onToggleSelect,
  onSelectAll,
  onDeselectAll,
  onRemoveItem,
  onMoveUp,
  onMoveDown,
  onStartIngestion,
  onRetryItem,
  onClearQueue,
}: UploadQueueReviewProps) {
  if (queue.length === 0) return null

  const selectedCount = queue.filter((item) => item.selected).length
  const completedCount = queue.filter((item) => item.status === 'completed').length
  const failedCount = queue.filter((item) => item.status === 'failed').length
  const unprocessedCount = queue.filter(
    (item) => item.status === 'pending' || (item.status === 'failed' && !item.selected)
  ).length

  // Files eligible to process: selected and not yet completed
  const eligibleToProcess = queue.filter(
    (item) => item.selected && item.status !== 'completed'
  ).length

  return (
    <div
      role="region"
      aria-label="Upload queue"
      style={{
        background: '#FFFFFF',
        border: '1px solid #E5E7EB',
        borderRadius: '8px',
        padding: '8px 10px',
        marginBottom: '4px',
        boxShadow: '0 2px 10px rgba(0, 0, 0, 0.05)',
        fontFamily: "Inter, Arial, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        boxSizing: 'border-box',
        maxHeight: '260px',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {/* 1. Header Toolbar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          paddingBottom: '6px',
          borderBottom: '1px solid #F3F4F6',
          marginBottom: '6px',
          gap: '8px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', minWidth: 0 }}>
          <span style={{ fontSize: '11.5px', fontWeight: 600, color: '#111827' }}>
            Upload Queue
          </span>
          <span
            style={{
              fontSize: '10.5px',
              color: '#6B7280',
              background: '#F3F4F6',
              padding: '1px 6px',
              borderRadius: '10px',
            }}
          >
            {selectedCount} of {queue.length} selected
          </span>
        </div>

        {/* Selection & Queue Controls */}
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
          {!isProcessing && (
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
                onMouseEnter={(e) => (e.currentTarget.style.color = '#EF4444')}
                onMouseLeave={(e) => (e.currentTarget.style.color = '#9CA3AF')}
              >
                Clear
              </button>
            </>
          )}
        </div>
      </div>

      {/* 2. Scrollable File Cards List */}
      <div
        style={{
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
          gap: '4px',
          paddingRight: '2px',
          flex: 1,
        }}
      >
        {queue.map((item, index) => {
          const isActive = item.id === activeFileId
          const isPendingWaiting = isProcessing && item.status === 'pending'
          const isCompleted = item.status === 'completed'
          const isFailed = item.status === 'failed'

          return (
            <div
              key={item.id}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                padding: '5px 7px',
                borderRadius: '6px',
                background: isActive ? '#EFF6FF' : isCompleted ? '#F0FDF4' : '#F9FAFB',
                border: isActive
                  ? '1px solid #BFDBFE'
                  : isCompleted
                  ? '1px solid #BBF7D0'
                  : isFailed
                  ? '1px solid #FECACA'
                  : '1px solid #F3F4F6',
                opacity: isPendingWaiting ? 0.65 : 1,
                transition: 'all 0.15s ease',
                boxSizing: 'border-box',
              }}
            >
              {/* Checkbox (manual inclusion) */}
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
                  width: '13px',
                  height: '13px',
                }}
              />

              {/* Document Icon */}
              <svg
                width="13"
                height="13"
                viewBox="0 0 24 24"
                fill="none"
                stroke={isActive ? '#2563EB' : isCompleted ? '#16A34A' : '#6B7280'}
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                style={{ flexShrink: 0 }}
              >
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
              </svg>

              {/* File Info */}
              <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: '6px' }}>
                  <span
                    title={item.name}
                    style={{
                      fontSize: '11px',
                      fontWeight: 600,
                      color: isActive ? '#1E3A8A' : isFailed ? '#991B1B' : '#1F2937',
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                    }}
                  >
                    {item.name}
                  </span>
                  <span
                    style={{
                      fontSize: '9.5px',
                      color: '#9CA3AF',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {formatFileSize(item.size)}
                  </span>
                </div>

                {/* Subtitle / Status Detail */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px', marginTop: '1px' }}>
                  {item.status === 'uploading' && (
                    <span style={{ fontSize: '9.5px', color: '#2563EB', fontWeight: 500, display: 'inline-flex', alignItems: 'center', gap: '3px' }}>
                      <span
                        style={{
                          width: '8px',
                          height: '8px',
                          border: '1.5px solid #BFDBFE',
                          borderTopColor: '#2563EB',
                          borderRadius: '50%',
                          display: 'inline-block',
                          animation: 'spin 0.8s linear infinite',
                        }}
                      />
                      Processing…
                    </span>
                  )}
                  {item.status === 'completed' && (
                    <span style={{ fontSize: '9.5px', color: '#16A34A', fontWeight: 500 }}>
                      ✓ {item.skipped ? 'Already ingested' : `${item.chunksCreated ?? 0} chunks indexed`}
                    </span>
                  )}
                  {item.status === 'failed' && (
                    <span
                      title={item.errorMessage}
                      style={{
                        fontSize: '9.5px',
                        color: '#DC2626',
                        fontWeight: 500,
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                      }}
                    >
                      ✕ {item.errorMessage || 'Upload failed'}
                    </span>
                  )}
                  {item.status === 'pending' && (
                    <span style={{ fontSize: '9.5px', color: item.selected ? '#6B7280' : '#9CA3AF' }}>
                      {item.selected ? 'Queued' : 'Unchecked (will skip)'}
                    </span>
                  )}
                </div>
              </div>

              {/* Action Controls (Move Up/Down, Retry, Remove) */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '2px', flexShrink: 0 }}>
                {/* Individual Retry button on failure */}
                {isFailed && !isProcessing && (
                  <button
                    type="button"
                    onClick={() => onRetryItem(item.id)}
                    title="Retry this file"
                    style={{
                      background: '#FEF2F2',
                      border: '1px solid #FECACA',
                      color: '#DC2626',
                      borderRadius: '4px',
                      fontSize: '9.5px',
                      fontWeight: 600,
                      padding: '1px 5px',
                      cursor: 'pointer',
                    }}
                  >
                    Retry
                  </button>
                )}

                {/* Reordering Controls (disabled during active run or for active item) */}
                {!isProcessing && !isCompleted && (
                  <>
                    <button
                      type="button"
                      onClick={() => onMoveUp(index)}
                      disabled={index === 0}
                      title="Move up in order"
                      style={{
                        background: 'transparent',
                        border: 'none',
                        color: index === 0 ? '#E5E7EB' : '#9CA3AF',
                        fontSize: '11px',
                        cursor: index === 0 ? 'default' : 'pointer',
                        padding: '1px 3px',
                        lineHeight: 1,
                      }}
                      onMouseEnter={(e) => {
                        if (index !== 0) e.currentTarget.style.color = '#374151'
                      }}
                      onMouseLeave={(e) => {
                        if (index !== 0) e.currentTarget.style.color = '#9CA3AF'
                      }}
                    >
                      ▲
                    </button>
                    <button
                      type="button"
                      onClick={() => onMoveDown(index)}
                      disabled={index === queue.length - 1}
                      title="Move down in order"
                      style={{
                        background: 'transparent',
                        border: 'none',
                        color: index === queue.length - 1 ? '#E5E7EB' : '#9CA3AF',
                        fontSize: '11px',
                        cursor: index === queue.length - 1 ? 'default' : 'pointer',
                        padding: '1px 3px',
                        lineHeight: 1,
                      }}
                      onMouseEnter={(e) => {
                        if (index !== queue.length - 1) e.currentTarget.style.color = '#374151'
                      }}
                      onMouseLeave={(e) => {
                        if (index !== queue.length - 1) e.currentTarget.style.color = '#9CA3AF'
                      }}
                    >
                      ▼
                    </button>
                  </>
                )}

                {/* Remove button (disabled if currently uploading) */}
                {!isActive && (
                  <button
                    type="button"
                    onClick={() => onRemoveItem(item.id)}
                    disabled={isProcessing && isActive}
                    title="Remove from queue"
                    style={{
                      background: 'transparent',
                      border: 'none',
                      color: '#9CA3AF',
                      fontSize: '11px',
                      cursor: 'pointer',
                      padding: '1px 3px',
                      lineHeight: 1,
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.color = '#EF4444')}
                    onMouseLeave={(e) => (e.currentTarget.style.color = '#9CA3AF')}
                  >
                    ✕
                  </button>
                )}
              </div>
            </div>
          )
        })}
      </div>

      {/* 3. Footer Summary & Start Ingestion Button */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          paddingTop: '6px',
          marginTop: '6px',
          borderTop: '1px solid #F3F4F6',
          gap: '8px',
        }}
      >
        {/* Separate Counts: Completed, Failed, Unprocessed */}
        <div style={{ fontSize: '10px', color: '#6B7280', display: 'flex', gap: '5px' }}>
          {completedCount > 0 && (
            <span style={{ color: '#16A34A', fontWeight: 600 }}>{completedCount} completed</span>
          )}
          {failedCount > 0 && (
            <span style={{ color: '#DC2626', fontWeight: 600 }}>{failedCount} failed</span>
          )}
          {unprocessedCount > 0 && <span>{unprocessedCount} unprocessed</span>}
        </div>

        {/* Start / Ingest Button */}
        <button
          type="button"
          onClick={onStartIngestion}
          disabled={isProcessing || eligibleToProcess === 0 || !isDestinationReady}
          style={{
            background: isProcessing || eligibleToProcess === 0 || !isDestinationReady ? '#F3F4F6' : '#2563EB',
            color: isProcessing || eligibleToProcess === 0 || !isDestinationReady ? '#9CA3AF' : '#FFFFFF',
            border: 'none',
            borderRadius: '5px',
            fontSize: '11px',
            fontWeight: 600,
            padding: '4px 10px',
            cursor: isProcessing || eligibleToProcess === 0 || !isDestinationReady ? 'default' : 'pointer',
            transition: 'background 0.15s ease',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '4px',
          }}
        >
          {isProcessing ? (
            <>
              <span
                style={{
                  width: '9px',
                  height: '9px',
                  border: '1.5px solid #9CA3AF',
                  borderTopColor: '#4B5563',
                  borderRadius: '50%',
                  display: 'inline-block',
                  animation: 'spin 0.8s linear infinite',
                }}
              />
              <span>Ingesting…</span>
            </>
          ) : (
            <span>{!isDestinationReady ? 'Select Department' : `Ingest ${eligibleToProcess > 0 ? `${eligibleToProcess} Selected` : 'Files'}`}</span>
          )}
        </button>
      </div>
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
      onCompleted,
    }: {
      departmentCode: string
      token: string
      workspaceId?: string
      onCompleted?: () => void
    }) => {
      // Prevent duplicate submissions and concurrent runners
      if (isProcessingRef.current) return
      isProcessingRef.current = true
      setIsProcessing(true)

      try {
        // Iterate through queue snapshot in exact user-chosen order
        let queueSnapshot = [...queue]

        for (let i = 0; i < queueSnapshot.length; i++) {
          const item = queueSnapshot[i]
          // Process only checked files that are not already completed
          if (!item.selected || item.status === 'completed') {
            continue
          }

          setActiveFileId(item.id)
          setQueue((prev) =>
            prev.map((q) =>
              q.id === item.id ? { ...q, status: 'uploading', errorMessage: undefined } : q
            )
          )

          try {
            const resp = await uploadDocument(item.file, departmentCode, token, workspaceId)
            setQueue((prev) =>
              prev.map((q) =>
                q.id === item.id
                  ? {
                      ...q,
                      status: 'completed',
                      chunksCreated: resp.chunks_created,
                      skipped: resp.skipped,
                    }
                  : q
              )
            )
          } catch (err: any) {
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
            // Failure isolated: continue to next selected file
          }
        }

        onCompleted?.()
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
      }: {
        departmentCode: string
        token: string
        workspaceId?: string
        onCompleted?: () => void
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
        const resp = await uploadDocument(target.file, departmentCode, token, workspaceId)
        setQueue((prev) =>
          prev.map((q) =>
            q.id === id
              ? {
                  ...q,
                  status: 'completed',
                  chunksCreated: resp.chunks_created,
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
