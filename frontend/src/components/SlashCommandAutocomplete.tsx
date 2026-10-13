/**
 * Theme-Aware Slash Command Autocomplete Component & Hook
 *
 * Polished, restrained, production RAG Workspace UI.
 * Typography: Inter / Arial sans-serif.
 * Icons: Outline SVGs matching existing tools menu.
 * Surfaces: Compact, subtle borders, clean focus and hover states.
 */

import React, { useEffect, useRef, useState } from 'react'

export {
  SUPPORTED_SLASH_COMMANDS,
  SORTED_COMMANDS_BY_LENGTH,
  getActiveCommand,
  getFilteredCommands,
  getCommandProgressMessage,
  formatCommandResponse,
} from './slashCommandDefinitions'
export type { SlashCommandDef, SlashCommandIcon } from './slashCommandDefinitions'
import {
  SUPPORTED_SLASH_COMMANDS,
  getActiveCommand,
  getFilteredCommands,
  SlashCommandDef,
  SlashCommandIcon,
} from './slashCommandDefinitions'

/**
 * Clean outline SVG icon matching the RAG Workspace's navigation and toolbar icons
 */
export function SlashIcon({ icon }: { icon: SlashCommandIcon }) {
  const commonProps = {
    width: '13',
    height: '13',
    viewBox: '0 0 24 24',
    fill: 'none',
    stroke: 'currentColor',
    strokeWidth: '1.75',
    strokeLinecap: 'round' as const,
    strokeLinejoin: 'round' as const,
  }

  switch (icon) {
    case 'rem':
      return (
        <svg {...commonProps}>
          <path d="M12 2a7 7 0 0 0-7 7c0 2.38 1.19 4.47 3 5.74V17a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1v-2.26c1.81-1.27 3-3.36 3-5.74a7 7 0 0 0-7-7z" />
          <line x1="9" y1="21" x2="15" y2="21" />
        </svg>
      )
    case 'memory':
      return (
        <svg {...commonProps}>
          <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
          <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
        </svg>
      )
    case 'forget':
      return (
        <svg {...commonProps}>
          <polyline points="3 6 5 6 21 6" />
          <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
          <line x1="10" y1="11" x2="10" y2="17" />
          <line x1="14" y1="11" x2="14" y2="17" />
        </svg>
      )
    case 'tasks':
      return (
        <svg {...commonProps}>
          <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
          <line x1="16" y1="2" x2="16" y2="6" />
          <line x1="8" y1="2" x2="8" y2="6" />
          <line x1="3" y1="10" x2="21" y2="10" />
        </svg>
      )
    case 'tasks-today':
      return (
        <svg {...commonProps}>
          <circle cx="12" cy="12" r="10" />
          <polyline points="12 6 12 12 16 14" />
        </svg>
      )
    case 'tasks-overdue':
      return (
        <svg {...commonProps}>
          <circle cx="12" cy="12" r="10" />
          <line x1="12" y1="8" x2="12" y2="12" />
          <line x1="12" y1="16" x2="12.01" y2="16" />
        </svg>
      )
    case 'task-add':
      return (
        <svg {...commonProps}>
          <circle cx="12" cy="12" r="10" />
          <line x1="12" y1="8" x2="12" y2="16" />
          <line x1="8" y1="12" x2="16" y2="12" />
        </svg>
      )
    case 'task-done':
      return (
        <svg {...commonProps}>
          <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
          <polyline points="22 4 12 14.01 9 11.01" />
        </svg>
      )
    default:
      return null
  }
}

/**
 * Custom Hook for managing slash command autocomplete state in any chat composer.
 */
export function useSlashCommands({
  text,
  setText,
  textareaRef,
}: {
  text: string
  setText: (newVal: string) => void
  textareaRef: React.RefObject<HTMLTextAreaElement | null>
}) {
  const [isOpen, setIsOpen] = useState(false)
  const [selectedIndex, setSelectedIndex] = useState(0)
  const [isDismissedByEsc, setIsDismissedByEsc] = useState(false)

  const filteredCommands = getFilteredCommands(text)
  const { cmdDef: activeCommand, argumentText } = getActiveCommand(text)

  useEffect(() => {
    const trimmed = text.trimStart()
    if (!trimmed.startsWith('/')) {
      setIsOpen(false)
      setIsDismissedByEsc(false)
      return
    }

    if (isDismissedByEsc) return

    // If text already has command + space and it's a single-word command with argument (e.g. "/rem hello")
    const { cmdDef, isExactCommandWithTrailingSpace } = getActiveCommand(text)
    if (cmdDef && isExactCommandWithTrailingSpace && !cmdDef.command.startsWith('/task')) {
      setIsOpen(false)
      return
    }

    // Keep dropdown open to display filtered commands or "No matching commands" state
    setIsOpen(true)
    setSelectedIndex(0)
  }, [text, isDismissedByEsc, filteredCommands.length])

  const selectCommand = (cmd: SlashCommandDef) => {
    const newText = `${cmd.command} `
    setText(newText)
    setIsOpen(false)
    setIsDismissedByEsc(false)

    setTimeout(() => {
      if (textareaRef.current) {
        textareaRef.current.focus()
        textareaRef.current.setSelectionRange(newText.length, newText.length)
      }
    }, 0)
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>): boolean => {
    if (!isOpen) {
      return false
    }

    if (e.key === 'Escape') {
      e.preventDefault()
      setIsOpen(false)
      setIsDismissedByEsc(true)
      return true
    }

    if (filteredCommands.length === 0) {
      return false
    }

    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setSelectedIndex((prev) => (prev + 1) % filteredCommands.length)
      return true
    }

    if (e.key === 'ArrowUp') {
      e.preventDefault()
      setSelectedIndex((prev) => (prev - 1 + filteredCommands.length) % filteredCommands.length)
      return true
    }

    if (e.key === 'Enter' || e.key === 'Tab') {
      e.preventDefault()
      const selected = filteredCommands[selectedIndex]
      if (selected) {
        selectCommand(selected)
      }
      return true
    }

    return false
  }

  return {
    isOpen,
    setIsOpen,
    selectedIndex,
    setSelectedIndex,
    filteredCommands,
    activeCommand,
    argumentText,
    selectCommand,
    handleKeyDown,
  }
}

/**
 * Dropdown Menu for Slash Command Suggestions
 */
export function SlashCommandMenu({
  isOpen,
  commands,
  selectedIndex,
  onSelect,
  isLoading = false,
}: {
  isOpen: boolean
  commands: SlashCommandDef[]
  selectedIndex: number
  onSelect: (cmd: SlashCommandDef) => void
  isLoading?: boolean
}) {
  const menuRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!isOpen || !menuRef.current) return
    const activeEl = menuRef.current.querySelector(`[data-index="${selectedIndex}"]`)
    if (activeEl) {
      activeEl.scrollIntoView({ block: 'nearest' })
    }
  }, [selectedIndex, isOpen])

  if (!isOpen) return null

  // 1. Loading state (only shown when genuinely loading asynchronously)
  if (isLoading) {
    return (
      <div
        id="slash-command-menu"
        style={{
          position: 'absolute',
          bottom: 'calc(100% + 6px)',
          left: 0,
          right: 0,
          background: '#FFFFFF',
          border: '1px solid #E5E7EB',
          borderRadius: '8px',
          boxShadow: '0 4px 16px rgba(0, 0, 0, 0.08)',
          padding: '12px 14px',
          zIndex: 50,
          fontFamily: "Inter, Arial, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
          boxSizing: 'border-box',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#6B7280', fontSize: '11.5px' }}>
          <div
            style={{
              width: '12px',
              height: '12px',
              border: '2px solid #E5E7EB',
              borderTopColor: '#4B5563',
              borderRadius: '50%',
              animation: 'spin 0.8s linear infinite',
            }}
          />
          <span>Loading commands…</span>
        </div>
      </div>
    )
  }

  // 2. No-match state
  if (commands.length === 0) {
    return (
      <div
        role="listbox"
        aria-label="Slash commands"
        id="slash-command-menu"
        style={{
          position: 'absolute',
          bottom: 'calc(100% + 6px)',
          left: 0,
          right: 0,
          background: '#FFFFFF',
          border: '1px solid #E5E7EB',
          borderRadius: '8px',
          boxShadow: '0 4px 16px rgba(0, 0, 0, 0.08)',
          padding: '10px 12px',
          zIndex: 50,
          fontFamily: "Inter, Arial, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
          boxSizing: 'border-box',
        }}
      >
        <div style={{ fontWeight: 600, fontSize: '12px', color: '#111827', marginBottom: '2px' }}>
          No matching commands
        </div>
        <div style={{ fontSize: '11px', color: '#6B7280' }}>
          Try another command prefix.
        </div>
      </div>
    )
  }

  const memoryCommands = commands.filter((c) => c.category === 'Personal Memory')
  const taskCommands = commands.filter((c) => c.category === 'Tasks & Follow-ups')

  let globalIndex = 0

  return (
    <div
      ref={menuRef}
      role="listbox"
      aria-label="Slash commands"
      id="slash-command-menu"
      style={{
        position: 'absolute',
        bottom: 'calc(100% + 6px)',
        left: 0,
        right: 0,
        maxHeight: '230px',
        overflowY: 'auto',
        background: '#FFFFFF',
        border: '1px solid #E5E7EB',
        borderRadius: '8px',
        boxShadow: '0 4px 16px rgba(0, 0, 0, 0.08)',
        padding: '4px',
        zIndex: 50,
        fontFamily: "Inter, Arial, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        boxSizing: 'border-box',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '4px 8px 5px',
          borderBottom: '1px solid #F3F4F6',
          marginBottom: '3px',
        }}
      >
        <span
          style={{
            fontSize: '11px',
            fontWeight: 600,
            color: '#4B5563',
          }}
        >
          Commands
        </span>
        <span
          style={{
            fontSize: '10px',
            color: '#9CA3AF',
            fontFamily: "var(--font-mono, 'DM Mono', monospace)",
          }}
        >
          ↑↓ navigate · ↵ select · Esc dismiss
        </span>
      </div>

      {memoryCommands.length > 0 && (
        <div style={{ marginBottom: '2px' }}>
          <div
            style={{
              fontSize: '10px',
              fontWeight: 600,
              color: '#9CA3AF',
              textTransform: 'uppercase',
              letterSpacing: '0.04em',
              padding: '3px 8px 2px',
            }}
          >
            Personal Memory
          </div>
          {memoryCommands.map((cmd) => {
            const currentIndex = globalIndex++
            const isSelected = currentIndex === selectedIndex
            return (
              <SlashCommandOption
                key={cmd.command}
                cmd={cmd}
                index={currentIndex}
                isSelected={isSelected}
                onSelect={() => onSelect(cmd)}
              />
            )
          })}
        </div>
      )}

      {taskCommands.length > 0 && (
        <div>
          <div
            style={{
              fontSize: '10px',
              fontWeight: 600,
              color: '#9CA3AF',
              textTransform: 'uppercase',
              letterSpacing: '0.04em',
              padding: '3px 8px 2px',
            }}
          >
            Tasks & Follow-ups
          </div>
          {taskCommands.map((cmd) => {
            const currentIndex = globalIndex++
            const isSelected = currentIndex === selectedIndex
            return (
              <SlashCommandOption
                key={cmd.command}
                cmd={cmd}
                index={currentIndex}
                isSelected={isSelected}
                onSelect={() => onSelect(cmd)}
              />
            )
          })}
        </div>
      )}
    </div>
  )
}

function SlashCommandOption({
  cmd,
  index,
  isSelected,
  onSelect,
}: {
  cmd: SlashCommandDef
  index: number
  isSelected: boolean
  onSelect: () => void
}) {
  return (
    <div
      role="option"
      aria-selected={isSelected}
      data-index={index}
      id={`slash-cmd-${index}`}
      onClick={onSelect}
      style={{
        display: 'flex',
        alignItems: 'flex-start',
        gap: '9px',
        padding: '5px 8px',
        borderRadius: '6px',
        cursor: 'pointer',
        background: isSelected ? '#F3F4F6' : 'transparent',
        transition: 'background 0.1s ease',
        boxSizing: 'border-box',
      }}
      onMouseEnter={(e) => {
        if (!isSelected) e.currentTarget.style.background = '#F9FAFB'
      }}
      onMouseLeave={(e) => {
        if (!isSelected) e.currentTarget.style.background = 'transparent'
      }}
    >
      {/* 1. Leading outline icon in a fixed-width column */}
      <div
        style={{
          width: '16px',
          height: '16px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexShrink: 0,
          marginTop: '1px',
          color: isSelected ? '#374151' : '#6B7280',
        }}
      >
        <SlashIcon icon={cmd.icon} />
      </div>

      {/* 2. Primary text column: command name and muted args placeholder */}
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '1px',
          minWidth: 0,
          flex: 1,
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'baseline',
            gap: '6px',
            minWidth: 0,
            whiteSpace: 'nowrap',
          }}
        >
          <span
            style={{
              fontSize: '12.5px',
              fontWeight: 600,
              color: isSelected ? '#111827' : '#1F2937',
              fontFamily: "Inter, Arial, -apple-system, sans-serif",
              lineHeight: 1.25,
            }}
          >
            {cmd.command}
          </span>
          {cmd.args && (
            <span
              style={{
                fontSize: '11px',
                fontWeight: 400,
                color: '#9CA3AF',
                fontFamily: "Inter, Arial, -apple-system, sans-serif",
                lineHeight: 1.25,
              }}
            >
              {cmd.args}
            </span>
          )}
        </div>

        {/* 3. Description directly beneath the command name, on exactly one line */}
        <div
          style={{
            fontSize: '11px',
            color: '#6B7280',
            lineHeight: 1.3,
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            minWidth: 0,
          }}
          title={cmd.description}
        >
          {cmd.description}
        </div>
      </div>
    </div>
  )
}

/**
 * Command-Aware Input Guidance Banner
 *
 * Restrained, native visual styling for the recognized command prefix and arguments.
 */
export function SlashCommandGuidance({
  activeCommand,
  argumentText,
  onClear,
}: {
  activeCommand: SlashCommandDef | null
  argumentText: string
  onClear?: () => void
}) {
  if (!activeCommand) return null

  let guidanceText = ''
  if (activeCommand.requiresArg) {
    if (!argumentText.trim()) {
      guidanceText = `Enter ${activeCommand.args || 'argument'}`
      if (activeCommand.example) {
        guidanceText += ` (e.g. "${activeCommand.example.replace(activeCommand.command, '').trim()}")`
      }
    } else {
      guidanceText = `${argumentText.trim().length} char${argumentText.trim().length !== 1 ? 's' : ''} entered — press Enter to send`
    }
  } else {
    guidanceText = 'Press Enter to execute'
  }

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '3px 8px',
        marginBottom: '3px',
        background: '#F9FAFB',
        border: '1px solid #E5E7EB',
        borderRadius: '6px',
        fontSize: '11px',
        fontFamily: "Inter, Arial, -apple-system, sans-serif",
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', minWidth: 0 }}>
        <span
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '4px',
            background: '#FFFFFF',
            border: '1px solid #D1D5DB',
            borderRadius: '4px',
            padding: '1px 5px',
            color: '#1F2937',
            fontWeight: 600,
            fontSize: '11px',
          }}
        >
          <span style={{ color: '#4B5563', display: 'flex' }}>
            <SlashIcon icon={activeCommand.icon} />
          </span>
          <span>{activeCommand.command}</span>
        </span>
        <span
          style={{
            color: '#6B7280',
            fontSize: '11px',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {guidanceText}
        </span>
      </div>

      {onClear && (
        <button
          type="button"
          onClick={onClear}
          title="Clear command"
          style={{
            background: 'transparent',
            border: 'none',
            color: '#9CA3AF',
            cursor: 'pointer',
            fontSize: '11px',
            lineHeight: 1,
            padding: '2px 4px',
          }}
          onMouseEnter={(e) => (e.currentTarget.style.color = '#374151')}
          onMouseLeave={(e) => (e.currentTarget.style.color = '#9CA3AF')}
        >
          ✕
        </button>
      )}
    </div>
  )
}
