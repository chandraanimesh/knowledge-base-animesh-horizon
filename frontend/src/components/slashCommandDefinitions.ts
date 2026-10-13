/**
 * Slash Command Definitions and Predictable Prefix Matching Logic
 *
 * Supported Commands (confirmed with backend /chat in src/rag_app/api/routes.py):
 * - Personal Memory: /rem, /remember, /memory, /memories, /forget, /forget_memory
 * - Tasks & Follow-ups: /tasks, /tasks today, /tasks overdue, /task add, /task done
 */

export type SlashCommandIcon =
  | 'rem'
  | 'memory'
  | 'forget'
  | 'tasks'
  | 'tasks-today'
  | 'tasks-overdue'
  | 'task-add'
  | 'task-done'

export interface SlashCommandDef {
  command: string
  label: string
  args?: string
  description: string
  category: 'Personal Memory' | 'Tasks & Follow-ups'
  requiresArg: boolean
  example?: string
  icon: SlashCommandIcon
}

export const SUPPORTED_SLASH_COMMANDS: SlashCommandDef[] = [
  // Personal Memory Commands
  {
    command: '/rem',
    label: '/rem',
    args: '<text>',
    description: 'Save a personal preference or memory',
    category: 'Personal Memory',
    requiresArg: true,
    example: '/rem I prefer responses in concise bullet points',
    icon: 'rem',
  },
  {
    command: '/remember',
    label: '/remember',
    args: '<text>',
    description: 'Save a personal preference or memory (alias)',
    category: 'Personal Memory',
    requiresArg: true,
    example: '/remember Call me Alex and keep tone professional',
    icon: 'rem',
  },
  {
    command: '/memory',
    label: '/memory',
    description: 'List all active personal memories',
    category: 'Personal Memory',
    requiresArg: false,
    example: '/memory',
    icon: 'memory',
  },
  {
    command: '/memories',
    label: '/memories',
    description: 'List all active personal memories (alias)',
    category: 'Personal Memory',
    requiresArg: false,
    example: '/memories',
    icon: 'memory',
  },
  {
    command: '/forget',
    label: '/forget',
    args: '<memory_id or keyword>',
    description: 'Archive a memory by UUID or matching keyword',
    category: 'Personal Memory',
    requiresArg: true,
    example: '/forget bullet points',
    icon: 'forget',
  },
  {
    command: '/forget_memory',
    label: '/forget_memory',
    args: '<memory_id or keyword>',
    description: 'Archive a memory by UUID or keyword (alias)',
    category: 'Personal Memory',
    requiresArg: true,
    example: '/forget_memory 550e8400-e29b-41d4-a716-446655440000',
    icon: 'forget',
  },

  // Tasks & Follow-ups Commands
  {
    command: '/tasks',
    label: '/tasks',
    description: 'List all open follow-up tasks',
    category: 'Tasks & Follow-ups',
    requiresArg: false,
    example: '/tasks',
    icon: 'tasks',
  },
  {
    command: '/tasks today',
    label: '/tasks today',
    description: 'List follow-up tasks due today',
    category: 'Tasks & Follow-ups',
    requiresArg: false,
    example: '/tasks today',
    icon: 'tasks-today',
  },
  {
    command: '/tasks overdue',
    label: '/tasks overdue',
    description: 'List overdue follow-up tasks',
    category: 'Tasks & Follow-ups',
    requiresArg: false,
    example: '/tasks overdue',
    icon: 'tasks-overdue',
  },
  {
    command: '/task add',
    label: '/task add',
    args: '<description>',
    description: 'Create a new follow-up task',
    category: 'Tasks & Follow-ups',
    requiresArg: true,
    example: '/task add Review Phoenix contract by Friday',
    icon: 'task-add',
  },
  {
    command: '/task done',
    label: '/task done',
    args: '<task_id>',
    description: 'Mark a follow-up task as completed',
    category: 'Tasks & Follow-ups',
    requiresArg: true,
    example: '/task done 550e8400-e29b-41d4-a716-446655440000',
    icon: 'task-done',
  },
]

// Sorted so longer commands match first when parsing active input
export const SORTED_COMMANDS_BY_LENGTH: SlashCommandDef[] = [...SUPPORTED_SLASH_COMMANDS].sort(
  (a, b) => b.command.length - a.command.length
)

/**
 * Checks if the current input text has a recognized, selected slash command.
 */
export function getActiveCommand(text: string): {
  cmdDef: SlashCommandDef | null
  argumentText: string
  isExactCommandWithTrailingSpace: boolean
} {
  const trimmedLeading = text.trimStart()
  if (!trimmedLeading.startsWith('/')) {
    return { cmdDef: null, argumentText: '', isExactCommandWithTrailingSpace: false }
  }

  const lower = trimmedLeading.toLowerCase()
  for (const cmd of SORTED_COMMANDS_BY_LENGTH) {
    const cmdLower = cmd.command.toLowerCase()
    if (lower === cmdLower) {
      return { cmdDef: cmd, argumentText: '', isExactCommandWithTrailingSpace: false }
    }
    if (lower.startsWith(cmdLower + ' ')) {
      const arg = trimmedLeading.slice(cmd.command.length + 1)
      return { cmdDef: cmd, argumentText: arg, isExactCommandWithTrailingSpace: true }
    }
  }

  return { cmdDef: null, argumentText: '', isExactCommandWithTrailingSpace: false }
}

/**
 * Filters commands using predictable prefix matching.
 */
export function getFilteredCommands(text: string): SlashCommandDef[] {
  const trimmed = text.trimStart()
  if (!trimmed.startsWith('/')) return []

  const search = trimmed.toLowerCase()

  // If a command requiring args already has a space after it (e.g. "/rem something"),
  // autocomplete menu should close unless it's a prefix for multi-word commands (like "/task " or "/tasks ")
  const isMultiWordPrefix =
    search === '/task' ||
    search === '/task ' ||
    search === '/tasks' ||
    search === '/tasks '
  if (trimmed.includes(' ') && !isMultiWordPrefix) {
    const matching = SUPPORTED_SLASH_COMMANDS.filter((cmd) =>
      cmd.command.toLowerCase().startsWith(search)
    )
    return matching
  }

  // Predictable prefix matching:
  // Commands that strictly start with the search prefix
  const matching = SUPPORTED_SLASH_COMMANDS.filter((cmd) => {
    const cLower = cmd.command.toLowerCase()
    return cLower.startsWith(search)
  })

  // If search is "/task", prioritize "/task add" and "/task done" before "/tasks"
  if (search === '/task' || search === '/task ') {
    return matching.sort((a, b) => {
      const aIsTaskAction = a.command.startsWith('/task ')
      const bIsTaskAction = b.command.startsWith('/task ')
      if (aIsTaskAction && !bIsTaskAction) return -1
      if (!aIsTaskAction && bIsTaskAction) return 1
      return 0
    })
  }

  return matching
}

/**
 * Returns contextual progress message during slash command execution.
 */
export function getCommandProgressMessage(query: string): string {
  const trimmed = query.trim().toLowerCase()
  if (trimmed.startsWith('/memory') || trimmed.startsWith('/memories')) {
    return 'Retrieving saved memories…'
  }
  if (trimmed.startsWith('/tasks today')) {
    return "Loading today's tasks…"
  }
  if (trimmed.startsWith('/tasks overdue')) {
    return 'Loading overdue tasks…'
  }
  if (trimmed.startsWith('/task add')) {
    return 'Creating task…'
  }
  if (trimmed.startsWith('/task done')) {
    return 'Updating task…'
  }
  if (trimmed.startsWith('/tasks') || trimmed.startsWith('/task')) {
    return 'Loading tasks…'
  }
  if (trimmed.startsWith('/rem') || trimmed.startsWith('/remember')) {
    return 'Saving memory…'
  }
  if (trimmed.startsWith('/forget')) {
    return 'Archiving memory…'
  }
  return 'Retrieving department context & generating answer…'
}

/**
 * Formats backend slash command responses into polished, context-specific empty result states.
 */
export function formatCommandResponse(query: string, rawAnswer: string): string {
  const trimmed = query.trim().toLowerCase()

  // 1. Personal Memory Cleanup & Empty State
  if (trimmed.startsWith('/rem') || trimmed.startsWith('/remember')) {
    if (rawAnswer) {
      return rawAnswer.replace(/\n\n\*ID:\*\s*`[a-f0-9-]+`/gi, '').trim()
    }
  }

  if (trimmed.startsWith('/memory') || trimmed.startsWith('/memories')) {
    if (
      !rawAnswer ||
      rawAnswer.toLowerCase().includes('no active personal memories') ||
      rawAnswer.toLowerCase().includes('no personal memories')
    ) {
      return '**No saved memories yet**\n\nUse `/rem` followed by something you want to remember.'
    }
  }

  // 2. Tasks Due Today Empty State
  if (trimmed.startsWith('/tasks today')) {
    if (
      !rawAnswer ||
      rawAnswer.toLowerCase().includes('no follow-ups due today') ||
      rawAnswer.toLowerCase().includes('no tasks due today')
    ) {
      return '**No tasks due today**'
    }
  }

  // 3. Overdue Tasks Empty State
  if (trimmed.startsWith('/tasks overdue')) {
    if (
      !rawAnswer ||
      rawAnswer.toLowerCase().includes('no overdue follow-ups') ||
      rawAnswer.toLowerCase().includes('no overdue tasks')
    ) {
      return '**No overdue tasks**'
    }
  }

  // 4. All Tasks Empty State
  if (trimmed === '/tasks' || trimmed === '/task' || trimmed.startsWith('/tasks list') || trimmed.startsWith('/task list')) {
    if (
      !rawAnswer ||
      rawAnswer.toLowerCase().includes('no tasks or follow-ups recorded') ||
      rawAnswer.toLowerCase().includes('no open tasks')
    ) {
      return '**No open tasks**'
    }
  }

  return rawAnswer
}

