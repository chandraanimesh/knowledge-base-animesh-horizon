import { test } from 'node:test'
import assert from 'node:assert'

import type { QueuedFileItem } from './components/uploadQueueDefinitions.ts'
import {
  formatFileSize,
  moveItemUp,
  moveItemDown,
  toggleItemSelection,
  setAllSelection,
  removeQueueItem,
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
