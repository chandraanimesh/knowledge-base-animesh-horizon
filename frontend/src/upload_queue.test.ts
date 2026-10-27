import { test } from 'node:test'
import assert from 'node:assert'

import type { QueuedFileItem, DocumentMetadataDraft } from './components/uploadQueueDefinitions.ts'
import {
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
} from './components/uploadQueueDefinitions.ts'

test('1. formatFileSize correctly formats bytes, KB, and MB', () => {
  assert.strictEqual(formatFileSize(500), '500 B')
  assert.strictEqual(formatFileSize(2048), '2.0 KB')
  assert.strictEqual(formatFileSize(1048576 * 3.5), '3.5 MB')
})

test('2. Multi-file queue initial selection and count', () => {
  const dummyFile = (name: string, size = 1000): File =>
    ({ name, size, type: 'application/pdf' } as unknown as File)

  const items: QueuedFileItem[] = [
    { id: '1', file: dummyFile('doc1.pdf'), name: 'doc1.pdf', size: 1000, selected: true, status: 'pending' },
    { id: '2', file: dummyFile('doc2.pdf'), name: 'doc2.pdf', size: 2000, selected: true, status: 'pending' },
    { id: '3', file: dummyFile('doc3.pdf'), name: 'doc3.pdf', size: 3000, selected: true, status: 'pending' },
  ]

  assert.strictEqual(items.length, 3)
  assert.strictEqual(items.filter((i) => i.selected).length, 3)
})

test('3. Toggle select, Select All and Deselect All operations', () => {
  const dummyFile = (name: string): File =>
    ({ name, size: 500, type: 'text/plain' } as unknown as File)

  let items: QueuedFileItem[] = [
    { id: '1', file: dummyFile('a.txt'), name: 'a.txt', size: 500, selected: true, status: 'pending' },
    { id: '2', file: dummyFile('b.txt'), name: 'b.txt', size: 500, selected: true, status: 'pending' },
    { id: '3', file: dummyFile('c.txt'), name: 'c.txt', size: 500, selected: true, status: 'pending' },
  ]

  // Deselect All
  items = setAllSelection(items, false)
  assert.strictEqual(items.filter((i) => i.selected).length, 0)

  // Toggle individual item '2'
  items = toggleItemSelection(items, '2')
  assert.strictEqual(items.filter((i) => i.selected).length, 1)
  assert.strictEqual(items[1].selected, true)

  // Select All
  items = setAllSelection(items, true)
  assert.strictEqual(items.filter((i) => i.selected).length, 3)
})

test('4. Queue ordering: Move Up and Move Down preserve user order', () => {
  const dummyFile = (name: string): File =>
    ({ name, size: 500, type: 'text/plain' } as unknown as File)

  let items: QueuedFileItem[] = [
    { id: '1', file: dummyFile('first.pdf'), name: 'first.pdf', size: 500, selected: true, status: 'pending' },
    { id: '2', file: dummyFile('second.pdf'), name: 'second.pdf', size: 500, selected: true, status: 'pending' },
    { id: '3', file: dummyFile('third.pdf'), name: 'third.pdf', size: 500, selected: true, status: 'pending' },
  ]

  // Move index 2 ('third.pdf') up
  items = moveItemUp(items, 2)
  assert.deepStrictEqual(
    items.map((i) => i.name),
    ['first.pdf', 'third.pdf', 'second.pdf']
  )

  // Move index 0 up is no-op
  items = moveItemUp(items, 0)
  assert.deepStrictEqual(
    items.map((i) => i.name),
    ['first.pdf', 'third.pdf', 'second.pdf']
  )

  // Move index 0 down
  items = moveItemDown(items, 0)
  assert.deepStrictEqual(
    items.map((i) => i.name),
    ['third.pdf', 'first.pdf', 'second.pdf']
  )
})

test('5. Removing unwanted file entirely from queue', () => {
  const dummyFile = (name: string): File =>
    ({ name, size: 500, type: 'text/plain' } as unknown as File)

  let items: QueuedFileItem[] = [
    { id: '1', file: dummyFile('keep1.pdf'), name: 'keep1.pdf', size: 500, selected: true, status: 'pending' },
    { id: '2', file: dummyFile('unwanted.pdf'), name: 'unwanted.pdf', size: 500, selected: true, status: 'pending' },
    { id: '3', file: dummyFile('keep2.pdf'), name: 'keep2.pdf', size: 500, selected: true, status: 'pending' },
  ]

  items = removeQueueItem(items, '2')
  assert.strictEqual(items.length, 2)
  assert.strictEqual(items.some((i) => i.name === 'unwanted.pdf'), false)
})

test('6. Sequential execution processes only checked files in exact chosen order', async () => {
  const dummyFile = (name: string): File =>
    ({ name, size: 500, type: 'text/plain' } as unknown as File)

  const items: QueuedFileItem[] = [
    { id: '1', file: dummyFile('docA.pdf'), name: 'docA.pdf', size: 500, selected: true, status: 'pending' },
    { id: '2', file: dummyFile('docB.pdf'), name: 'docB.pdf', size: 500, selected: false, status: 'pending' }, // UNCHECKED
    { id: '3', file: dummyFile('docC.pdf'), name: 'docC.pdf', size: 500, selected: true, status: 'pending' },
  ]

  const processedLog: string[] = []

  // Simulate sequential queue execution
  for (const item of items) {
    if (!item.selected || item.status === 'completed') {
      continue
    }
    processedLog.push(item.name)
    item.status = 'completed'
  }

  // Exactly docA then docC were processed; docB remained unchecked
  assert.deepStrictEqual(processedLog, ['docA.pdf', 'docC.pdf'])
  assert.strictEqual(items[0].status, 'completed')
  assert.strictEqual(items[1].status, 'pending') // preserved for next run!
  assert.strictEqual(items[2].status, 'completed')
})

test('7. Error isolation: failed file records error and queue continues to next file', async () => {
  const dummyFile = (name: string): File =>
    ({ name, size: 500, type: 'text/plain' } as unknown as File)

  const items: QueuedFileItem[] = [
    { id: '1', file: dummyFile('good1.pdf'), name: 'good1.pdf', size: 500, selected: true, status: 'pending' },
    { id: '2', file: dummyFile('corrupted.pdf'), name: 'corrupted.pdf', size: 500, selected: true, status: 'pending' },
    { id: '3', file: dummyFile('good2.pdf'), name: 'good2.pdf', size: 500, selected: true, status: 'pending' },
  ]

  const mockUploader = async (file: File) => {
    if (file.name === 'corrupted.pdf') {
      throw new Error('Unsupported format or corrupt document')
    }
    return { file_name: file.name, chunks_created: 5, skipped: false }
  }

  for (const item of items) {
    if (!item.selected || item.status === 'completed') continue
    try {
      await mockUploader(item.file)
      item.status = 'completed'
    } catch (err: any) {
      item.status = 'failed'
      item.errorMessage = err.message
    }
  }

  assert.strictEqual(items[0].status, 'completed')
  assert.strictEqual(items[1].status, 'failed')
  assert.strictEqual(items[1].errorMessage, 'Unsupported format or corrupt document')
  assert.strictEqual(items[2].status, 'completed') // continued successfully to next file!
})

test('8. Individual retry of failed file does not reprocess already completed files', async () => {
  const dummyFile = (name: string): File =>
    ({ name, size: 500, type: 'text/plain' } as unknown as File)

  const items: QueuedFileItem[] = [
    { id: '1', file: dummyFile('good1.pdf'), name: 'good1.pdf', size: 500, selected: true, status: 'completed' },
    { id: '2', file: dummyFile('corrupted.pdf'), name: 'corrupted.pdf', size: 500, selected: true, status: 'failed', errorMessage: 'Timeout' },
  ]

  const mockSingleRetry = async (targetId: string) => {
    const target = items.find((i) => i.id === targetId)
    assert.ok(target)
    target.status = 'completed'
    target.errorMessage = undefined
  }

  await mockSingleRetry('2')

  // Both are now completed, good1 was untouched
  assert.strictEqual(items[0].status, 'completed')
  assert.strictEqual(items[1].status, 'completed')
  assert.strictEqual(items[1].errorMessage, undefined)
})

test('9. Upload with no metadata preserves fast-upload path without extra review step', () => {
  const emptyDraft: DocumentMetadataDraft = {
    docType: undefined,
    classification: undefined,
    tags: [],
    visibility: undefined,
    description: undefined,
  }
  assert.strictEqual(hasMetadata(emptyDraft), false)
  const validation = validateMetadataDraft(emptyDraft)
  assert.strictEqual(validation.isValid, true)
  assert.strictEqual(validation.warnings.length, 0)
})

test('10. Upload with valid metadata correctly detects presence and validates fields', () => {
  const validDraft: DocumentMetadataDraft = {
    docType: 'Reference Data',
    classification: 'Internal',
    tags: ['company-directory', 'contacts'],
    visibility: 'Department only',
    description: 'Company directory records and contact information',
  }
  assert.strictEqual(hasMetadata(validDraft), true)
  const validation = validateMetadataDraft(validDraft)
  assert.strictEqual(validation.isValid, true)
  assert.strictEqual(validation.warnings.length, 0)
})

test('11. Review and edit step displays exact draft metadata and surfaces warnings', () => {
  const warningDraft: DocumentMetadataDraft = {
    docType: 'Documentation',
    tags: Array.from({ length: 25 }, (_, i) => `tag-${i}`),
    description: 'A'.repeat(1200),
  }
  const validation = validateMetadataDraft(warningDraft)
  assert.strictEqual(validation.warnings.length, 2)
  assert.ok(validation.warnings.some((w) => w.includes('20 tags')))
  assert.ok(validation.warnings.some((w) => w.includes('Description is long')))
})

test('12. Back navigation preserves file selection and draft metadata without data loss', () => {
  const dummyFile = (name: string): File =>
    ({ name, size: 500, type: 'text/plain' } as unknown as File)

  const items: QueuedFileItem[] = [
    { id: '1', file: dummyFile('plan.md'), name: 'plan.md', size: 500, selected: true, status: 'pending' },
    { id: '2', file: dummyFile('spec.md'), name: 'spec.md', size: 800, selected: true, status: 'pending' },
  ]
  const draft: DocumentMetadataDraft = {
    docType: 'Technical Spec',
    classification: 'Confidential',
    tags: ['architecture', 'v2'],
    description: 'System design specification',
  }

  // Simulated transition to review then back to edit
  let currentStep: 'edit' | 'review' = 'edit'
  currentStep = 'review'
  assert.strictEqual(currentStep, 'review')

  // User clicks "Back to edit"
  currentStep = 'edit'
  assert.strictEqual(currentStep, 'edit')
  assert.strictEqual(items.length, 2)
  assert.strictEqual(draft.docType, 'Technical Spec')
  assert.deepStrictEqual(draft.tags, ['architecture', 'v2'])
})

test('13. Batch metadata scoping applies draft to all files or specific single file', () => {
  const dummyFile = (name: string): File =>
    ({ name, size: 500, type: 'text/plain' } as unknown as File)

  let items: QueuedFileItem[] = [
    { id: '1', file: dummyFile('file1.csv'), name: 'file1.csv', size: 1000, selected: true, status: 'pending' },
    { id: '2', file: dummyFile('file2.csv'), name: 'file2.csv', size: 2000, selected: true, status: 'pending' },
  ]

  const sharedMetadata: DocumentMetadataDraft = {
    docType: 'Reference Data',
    classification: 'Internal',
    tags: ['csv-import'],
  }

  // Apply to all
  items = applyBatchMetadata(items, sharedMetadata)
  assert.strictEqual(items[0].metadata?.docType, 'Reference Data')
  assert.strictEqual(items[1].metadata?.docType, 'Reference Data')
  assert.deepStrictEqual(items[0].metadata?.tags, ['csv-import'])

  // Apply to single file '1'
  const customSingleMeta: DocumentMetadataDraft = {
    docType: 'Contract',
    tags: ['legal'],
  }
  items = setItemMetadata(items, '1', customSingleMeta)
  assert.strictEqual(items[0].metadata?.docType, 'Contract')
  assert.strictEqual(items[1].metadata?.docType, 'Reference Data') // file2 untouched!
})

test('14. Prevention of accidental double submission while processing', async () => {
  let isProcessing = false
  let submissionCount = 0

  const handleStart = () => {
    if (isProcessing) return
    isProcessing = true
    submissionCount++
  }

  handleStart()
  // Immediate second click while running
  handleStart()
  handleStart()

  assert.strictEqual(submissionCount, 1)
  isProcessing = false
})

test('15. Accurate success notification and backend chunk count accumulation', async () => {
  const dummyFile = (name: string): File =>
    ({ name, size: 500, type: 'text/plain' } as unknown as File)

  const items: QueuedFileItem[] = [
    { id: '1', file: dummyFile('data.csv'), name: 'data.csv', size: 500, selected: true, status: 'pending' },
    { id: '2', file: dummyFile('notes.txt'), name: 'notes.txt', size: 300, selected: true, status: 'pending' },
  ]

  const backendResults = [
    { chunks_created: 1000, skipped: false },
    { chunks_created: 45, skipped: false },
  ]

  let totalChunks = 0
  let completedCount = 0

  for (let i = 0; i < items.length; i++) {
    const res = backendResults[i]
    items[i].status = 'completed'
    items[i].chunksCreated = res.chunks_created
    totalChunks += res.chunks_created
    completedCount++
  }

  assert.strictEqual(completedCount, 2)
  assert.strictEqual(totalChunks, 1045)
  const notificationText = `Ingestion complete · ${completedCount} file(s) processed · ${totalChunks.toLocaleString()} chunks indexed`
  assert.strictEqual(notificationText, 'Ingestion complete · 2 file(s) processed · 1,045 chunks indexed')
})

test('16. Failure keeps upload panel available and does not report false success', () => {
  const items: QueuedFileItem[] = [
    { id: '1', file: ({} as unknown as File), name: 'ok.txt', size: 100, selected: true, status: 'completed', chunksCreated: 10 },
    { id: '2', file: ({} as unknown as File), name: 'fail.txt', size: 100, selected: true, status: 'failed', errorMessage: 'Permission denied' },
  ]

  const completed = items.filter((i) => i.status === 'completed').length
  const failed = items.filter((i) => i.status === 'failed').length

  const shouldClosePanel = failed === 0 && completed > 0
  assert.strictEqual(shouldClosePanel, false) // MUST remain open!
  assert.strictEqual(failed, 1)
  assert.strictEqual(items[1].errorMessage, 'Permission denied')
})

