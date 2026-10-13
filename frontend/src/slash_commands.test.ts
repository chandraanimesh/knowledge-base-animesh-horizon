import { test } from 'node:test'
import assert from 'node:assert'

import {
  SUPPORTED_SLASH_COMMANDS,
  getActiveCommand,
  getFilteredCommands,
  getCommandProgressMessage,
  formatCommandResponse,
} from './components/slashCommandDefinitions.ts'

test('1. Typing "/" returns all supported commands', () => {
  const commands = getFilteredCommands('/')
  assert.strictEqual(commands.length, 11)
  const names = commands.map((c) => c.command)
  assert.ok(names.includes('/rem'))
  assert.ok(names.includes('/remember'))
  assert.ok(names.includes('/memory'))
  assert.ok(names.includes('/memories'))
  assert.ok(names.includes('/forget'))
  assert.ok(names.includes('/forget_memory'))
  assert.ok(names.includes('/tasks'))
  assert.ok(names.includes('/tasks today'))
  assert.ok(names.includes('/tasks overdue'))
  assert.ok(names.includes('/task add'))
  assert.ok(names.includes('/task done'))
})

test('2. Typing "/rem" filters suggestions to /rem and /remember', () => {
  const commands = getFilteredCommands('/rem')
  const names = commands.map((c) => c.command)
  assert.ok(names.includes('/rem'))
  assert.ok(names.includes('/remember'))
  assert.ok(!names.includes('/tasks'))
  assert.ok(!names.includes('/forget'))
})

test('3. Typing "/for" filters suggestions to /forget and /forget_memory', () => {
  const commands = getFilteredCommands('/for')
  const names = commands.map((c) => c.command)
  assert.strictEqual(names.length, 2)
  assert.ok(names.includes('/forget'))
  assert.ok(names.includes('/forget_memory'))
})

test('4. "/tasks today" is clearly distinguishable from "/tasks" and "/tasks overdue"', () => {
  const tasksFiltered = getFilteredCommands('/tasks')
  const taskNames = tasksFiltered.map((c) => c.command)
  assert.ok(taskNames.includes('/tasks'))
  assert.ok(taskNames.includes('/tasks today'))
  assert.ok(taskNames.includes('/tasks overdue'))

  const todayFiltered = getFilteredCommands('/tasks t')
  assert.strictEqual(todayFiltered.length, 1)
  assert.strictEqual(todayFiltered[0].command, '/tasks today')
})

test('5. Subcommand filtering for "/task a" and "/task d"', () => {
  const addFiltered = getFilteredCommands('/task a')
  assert.strictEqual(addFiltered.length, 1)
  assert.strictEqual(addFiltered[0].command, '/task add')

  const doneFiltered = getFilteredCommands('/task d')
  assert.strictEqual(doneFiltered.length, 1)
  assert.strictEqual(doneFiltered[0].command, '/task done')
})

test('6. Pressing space after "/rem" preserves argument area and recognizes active command', () => {
  const active = getActiveCommand('/rem I prefer concise responses')
  assert.ok(active.cmdDef !== null)
  assert.strictEqual(active.cmdDef?.command, '/rem')
  assert.strictEqual(active.argumentText, 'I prefer concise responses')
  assert.strictEqual(active.isExactCommandWithTrailingSpace, true)
})

test('7. Selecting "/task add" leaves the composer ready for task text', () => {
  const active = getActiveCommand('/task add Deploy custom SSO integration')
  assert.ok(active.cmdDef !== null)
  assert.strictEqual(active.cmdDef?.command, '/task add')
  assert.strictEqual(active.argumentText, 'Deploy custom SSO integration')
  assert.strictEqual(active.cmdDef?.requiresArg, true)
})

test('8. Non-argument commands correctly indicate no arguments needed', () => {
  const activeTasks = getActiveCommand('/tasks today')
  assert.ok(activeTasks.cmdDef !== null)
  assert.strictEqual(activeTasks.cmdDef?.command, '/tasks today')
  assert.strictEqual(activeTasks.cmdDef?.requiresArg, false)

  const activeMem = getActiveCommand('/memory')
  assert.ok(activeMem.cmdDef !== null)
  assert.strictEqual(activeMem.cmdDef?.command, '/memory')
  assert.strictEqual(activeMem.cmdDef?.requiresArg, false)
})

test('9. Ordinary messages and slashes in normal text do not trigger slash commands', () => {
  assert.strictEqual(getFilteredCommands('What is 10/2?').length, 0)
  assert.strictEqual(getFilteredCommands('Please review document/contract').length, 0)
  assert.strictEqual(getFilteredCommands('').length, 0)

  const activeNonCmd = getActiveCommand('What is 10/2?')
  assert.strictEqual(activeNonCmd.cmdDef, null)
})

test('10. Command definitions contain required design metadata (icons, descriptions, examples)', () => {
  for (const cmd of SUPPORTED_SLASH_COMMANDS) {
    assert.ok(cmd.command.startsWith('/'))
    assert.ok(cmd.description.length > 0)
    assert.ok(cmd.icon.length > 0)
    assert.ok(cmd.category === 'Personal Memory' || cmd.category === 'Tasks & Follow-ups')
    if (cmd.requiresArg) {
      assert.ok(cmd.args !== undefined)
      assert.ok(cmd.example !== undefined)
    }
  }
})

test('11. Prefix matching for "/r" prioritizes /rem and /remember, and NEVER includes /memories', () => {
  const rCommands = getFilteredCommands('/r')
  const names = rCommands.map((c) => c.command)
  assert.strictEqual(names.length, 2)
  assert.ok(names.includes('/rem'))
  assert.ok(names.includes('/remember'))
  assert.ok(!names.includes('/memories'))
  assert.ok(!names.includes('/memory'))
})

test('12. Prefix matching for "/m" shows /memory and /memories', () => {
  const mCommands = getFilteredCommands('/m')
  const names = mCommands.map((c) => c.command)
  assert.strictEqual(names.length, 2)
  assert.ok(names.includes('/memory'))
  assert.ok(names.includes('/memories'))
})

test('13. Prefix matching for "/fo" shows /forget and /forget_memory', () => {
  const foCommands = getFilteredCommands('/fo')
  const names = foCommands.map((c) => c.command)
  assert.strictEqual(names.length, 2)
  assert.ok(names.includes('/forget'))
  assert.ok(names.includes('/forget_memory'))
})

test('14. Prefix matching for "/ta" shows all 5 task commands', () => {
  const taCommands = getFilteredCommands('/ta')
  const names = taCommands.map((c) => c.command)
  assert.strictEqual(names.length, 5)
  assert.ok(names.includes('/tasks'))
  assert.ok(names.includes('/tasks today'))
  assert.ok(names.includes('/tasks overdue'))
  assert.ok(names.includes('/task add'))
  assert.ok(names.includes('/task done'))
})

test('15. Prefix matching for "/task" prioritizes /task add and /task done before /tasks', () => {
  const taskCommands = getFilteredCommands('/task')
  const names = taskCommands.map((c) => c.command)
  assert.strictEqual(names.length, 5)
  // /task add and /task done must be prioritized at the beginning
  assert.ok(names.indexOf('/task add') < names.indexOf('/tasks'))
  assert.ok(names.indexOf('/task done') < names.indexOf('/tasks'))
  assert.strictEqual(names[0], '/task add')
  assert.strictEqual(names[1], '/task done')
})

test('16. Contextual progress feedback messages match specifications', () => {
  assert.strictEqual(getCommandProgressMessage('/memory'), 'Retrieving saved memories…')
  assert.strictEqual(getCommandProgressMessage('/memories'), 'Retrieving saved memories…')
  assert.strictEqual(getCommandProgressMessage('/tasks today'), "Loading today's tasks…")
  assert.strictEqual(getCommandProgressMessage('/tasks overdue'), 'Loading overdue tasks…')
  assert.strictEqual(getCommandProgressMessage('/task add Review contract'), 'Creating task…')
  assert.strictEqual(getCommandProgressMessage('/task done 123'), 'Updating task…')
  assert.strictEqual(getCommandProgressMessage('/tasks'), 'Loading tasks…')
  assert.strictEqual(getCommandProgressMessage('/rem Prefer concise bullets'), 'Saving memory…')
  assert.strictEqual(getCommandProgressMessage('/remember Keep professional'), 'Saving memory…')
  assert.strictEqual(getCommandProgressMessage('/forget 123'), 'Archiving memory…')
  assert.strictEqual(getCommandProgressMessage('Normal question'), 'Retrieving department context & generating answer…')
})

test('17. Empty result state formatting matches specifications', () => {
  // Personal Memory empty state
  const memEmpty = formatCommandResponse('/memory', '🧠 You currently have no active personal memories saved.')
  assert.strictEqual(memEmpty, '**No saved memories yet**\n\nUse `/rem` followed by something you want to remember.')

  // Tasks empty states
  const tasksEmpty = formatCommandResponse('/tasks', '📋 You currently have no tasks or follow-ups recorded.')
  assert.strictEqual(tasksEmpty, '**No open tasks**')

  const todayEmpty = formatCommandResponse('/tasks today', '📋 You have no follow-ups due today.')
  assert.strictEqual(todayEmpty, '**No tasks due today**')

  const overdueEmpty = formatCommandResponse('/tasks overdue', '✅ You have no overdue follow-ups.')
  assert.strictEqual(overdueEmpty, '**No overdue tasks**')

  // Non-empty responses remain untouched
  const normalAns = 'Here are your active memories: \n- Memory 1'
  assert.strictEqual(formatCommandResponse('/memory', normalAns), normalAns)
})

test('18. Unmatched slash prefix returns empty suggestions array to trigger No-match UI state', () => {
  const unmatched = getFilteredCommands('/xyz')
  assert.strictEqual(unmatched.length, 0)
})

test('19. /rem response strips raw UUID ID line while preserving content and scope', () => {
  const rawWithId = '🧠 **Memory saved** (global profile):\n> "I prefer concise bullet points"\n\n*ID:* `b4e58f27-3136-4076-a496-e260907ad53e`'
  const cleaned = formatCommandResponse('/rem I prefer concise bullet points', rawWithId)
  assert.strictEqual(cleaned, '🧠 **Memory saved** (global profile):\n> "I prefer concise bullet points"')
  assert.ok(!cleaned.includes('b4e58f27-3136-4076-a496-e260907ad53e'))
  assert.ok(!cleaned.includes('*ID:*'))
})



