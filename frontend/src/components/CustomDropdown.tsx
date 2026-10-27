import React, { useState, useRef, useEffect } from 'react';

export interface DropdownOption {
  value: string;
  label: string;
  description?: string;
  disabled?: boolean;
  badge?: string;
  icon?: React.ReactNode;
}

interface CustomDropdownProps {
  value: string;
  onChange: (val: string) => void;
  options: DropdownOption[];
  disabled?: boolean;
  placeholder?: string;
  className?: string;
  style?: React.CSSProperties;
}

export default function CustomDropdown({
  value,
  onChange,
  options,
  disabled = false,
  placeholder = 'Select an option...',
  style,
}: CustomDropdownProps) {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const selectedOption = options.find((opt) => opt.value === value);

  // Close on outside click
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen]);

  // Close on Escape key
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape' && isOpen) {
        setIsOpen(false);
      }
    }
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen]);

  const handleSelect = (option: DropdownOption) => {
    if (option.disabled || disabled) return;
    onChange(option.value);
    setIsOpen(false);
  };

  return (
    <div
      ref={containerRef}
      style={{
        position: 'relative',
        width: '100%',
        fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
        userSelect: 'none',
        ...style,
      }}
    >
      {/* Trigger Button */}
      <button
        type="button"
        disabled={disabled}
        onClick={() => !disabled && setIsOpen(!isOpen)}
        style={{
          width: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '8px 12px',
          background: disabled ? '#f8fafc' : '#ffffff',
          border: isOpen ? '1px solid #3b82f6' : '1px solid #cbd5e1',
          borderRadius: '8px',
          boxShadow: isOpen
            ? '0 0 0 3px rgba(59, 130, 246, 0.15), 0 1px 2px rgba(0, 0, 0, 0.05)'
            : '0 1px 2px rgba(0, 0, 0, 0.04)',
          cursor: disabled ? 'not-allowed' : 'pointer',
          color: disabled ? '#94a3b8' : '#0f172a',
          fontSize: '13px',
          fontWeight: 500,
          transition: 'all 0.15s ease-in-out',
          textAlign: 'left',
          boxSizing: 'border-box',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', overflow: 'hidden' }}>
          {selectedOption?.icon && (
            <span style={{ fontSize: '14px', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
              {selectedOption.icon}
            </span>
          )}
          <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {selectedOption ? selectedOption.label : placeholder}
          </span>
        </div>

        {/* Animated Chevron */}
        <svg
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.2"
          strokeLinecap="round"
          strokeLinejoin="round"
          style={{
            color: '#64748b',
            transform: isOpen ? 'rotate(180deg)' : 'rotate(0deg)',
            transition: 'transform 0.2s ease',
            flexShrink: 0,
            marginLeft: '8px',
          }}
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>

      {/* Floating Menu with Deep Soft Shadow */}
      {isOpen && (
        <div
          style={{
            position: 'absolute',
            top: 'calc(100% + 5px)',
            left: 0,
            right: 0,
            background: '#ffffff',
            border: '1px solid #e2e8f0',
            borderRadius: '10px',
            boxShadow: '0 14px 30px -4px rgba(15, 23, 42, 0.14), 0 6px 12px -3px rgba(15, 23, 42, 0.08)',
            padding: '5px',
            maxHeight: '260px',
            overflowY: 'auto',
            zIndex: 9999,
            display: 'flex',
            flexDirection: 'column',
            gap: '2px',
          }}
        >
          {options.map((option) => {
            const isSelected = option.value === value;
            return (
              <div
                key={option.value}
                onClick={() => handleSelect(option)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '8px 10px',
                  borderRadius: '6px',
                  background: isSelected ? '#eff6ff' : 'transparent',
                  color: option.disabled ? '#94a3b8' : isSelected ? '#1d4ed8' : '#1e293b',
                  fontSize: '12.5px',
                  fontWeight: isSelected ? 600 : 400,
                  cursor: option.disabled ? 'not-allowed' : 'pointer',
                  opacity: option.disabled ? 0.6 : 1,
                  transition: 'background 0.12s ease',
                }}
                onMouseEnter={(e) => {
                  if (!option.disabled && !isSelected) {
                    e.currentTarget.style.background = '#f1f5f9';
                  }
                }}
                onMouseLeave={(e) => {
                  if (!isSelected) {
                    e.currentTarget.style.background = 'transparent';
                  }
                }}
              >
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    {option.icon && (
                      <span style={{ fontSize: '13px', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                        {option.icon}
                      </span>
                    )}
                    <span>{option.label}</span>
                  </div>
                  {option.description && (
                    <span style={{ fontSize: '10.5px', color: '#64748b', marginLeft: option.icon ? '19px' : '0' }}>
                      {option.description}
                    </span>
                  )}
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  {option.badge && (
                    <span
                      style={{
                        fontSize: '10px',
                        fontWeight: 600,
                        padding: '2px 6px',
                        borderRadius: '4px',
                        background: option.disabled ? '#f1f5f9' : '#e0f2fe',
                        color: option.disabled ? '#64748b' : '#0369a1',
                        border: option.disabled ? '1px solid #e2e8f0' : '1px solid #bae6fd',
                      }}
                    >
                      {option.badge}
                    </span>
                  )}
                  {isSelected && (
                    <span style={{ color: '#2563eb', fontWeight: 700, fontSize: '13px' }}>✓</span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
