/**
 * Upload Queue Definitions and Utilities
 */

export type UploadQueueStatus = 'pending' | 'uploading' | 'completed' | 'failed'

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
}

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
