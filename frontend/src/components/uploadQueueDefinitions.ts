/**
 * Upload Queue Definitions and Utilities
 */

export type UploadQueueStatus = 'pending' | 'uploading' | 'completed' | 'failed'

export interface DocumentMetadataDraft {
  docType?: string
  classification?: string
  tags: string[]
  visibility?: string
  description?: string
  isSuggested?: boolean
}

export interface QueuedFileItem {
  id: string
  file: File
  name: string
  size: number
  selected: boolean
  status: UploadQueueStatus
  errorMessage?: string
  chunksCreated?: number
  skipped?: boolean
  metadata?: DocumentMetadataDraft
}

export const ALLOWED_DOC_TYPES = [
  'Reference Data',
  'Policy',
  'Documentation',
  'Report',
  'Technical Spec',
  'Contract',
  'Manual',
  'Other',
] as const

export const ALLOWED_CLASSIFICATIONS = [
  'Internal',
  'Public',
  'Confidential',
  'Restricted',
] as const

export const ALLOWED_VISIBILITY_OPTIONS = [
  'Department only',
  'Workspace',
  'Organization-wide',
] as const

export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export function moveItemUp(list: QueuedFileItem[], index: number): QueuedFileItem[] {
  if (index <= 0 || index >= list.length) return list
  const copy = [...list]
  const temp = copy[index - 1]
  copy[index - 1] = copy[index]
  copy[index] = temp
  return copy
}

export function moveItemDown(list: QueuedFileItem[], index: number): QueuedFileItem[] {
  if (index < 0 || index >= list.length - 1) return list
  const copy = [...list]
  const temp = copy[index + 1]
  copy[index + 1] = copy[index]
  copy[index] = temp
  return copy
}

export function toggleItemSelection(list: QueuedFileItem[], id: string): QueuedFileItem[] {
  return list.map((item) => (item.id === id ? { ...item, selected: !item.selected } : item))
}

export function setAllSelection(list: QueuedFileItem[], selected: boolean): QueuedFileItem[] {
  return list.map((item) => ({ ...item, selected }))
}

export function removeQueueItem(list: QueuedFileItem[], id: string): QueuedFileItem[] {
  return list.filter((item) => item.id !== id)
}

export function hasMetadata(meta?: DocumentMetadataDraft): boolean {
  if (!meta) return false
  return Boolean(
    (meta.docType && meta.docType.trim()) ||
    (meta.classification && meta.classification.trim()) ||
    (meta.tags && meta.tags.length > 0) ||
    (meta.visibility && meta.visibility.trim()) ||
    (meta.description && meta.description.trim())
  )
}

export function validateMetadataDraft(meta?: DocumentMetadataDraft): { isValid: boolean; warnings: string[] } {
  const warnings: string[] = []
  if (!meta) return { isValid: true, warnings }

  if (meta.tags && meta.tags.length > 20) {
    warnings.push('Tag count exceeds recommended 20 tags.')
  }
  if (meta.description && meta.description.length > 1000) {
    warnings.push('Description is long and may be summarized.')
  }
  return { isValid: true, warnings }
}

export function applyBatchMetadata(list: QueuedFileItem[], meta: DocumentMetadataDraft): QueuedFileItem[] {
  return list.map((item) => ({
    ...item,
    metadata: {
      ...meta,
      tags: [...(meta.tags || [])],
    },
  }))
}

export function setItemMetadata(
  list: QueuedFileItem[],
  id: string,
  meta: DocumentMetadataDraft
): QueuedFileItem[] {
  return list.map((item) =>
    item.id === id
      ? {
          ...item,
          metadata: {
            ...meta,
            tags: [...(meta.tags || [])],
          },
        }
      : item
  )
}
