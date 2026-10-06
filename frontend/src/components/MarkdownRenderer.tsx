import React from 'react';

interface MarkdownRendererProps {
  content: string;
  className?: string;
  style?: React.CSSProperties;
}

/**
 * Parses inline markdown:
 * - Bold: **text** or __text__
 * - Italic: *text* or _text_
 * - Inline code: `code`
 * - Citations: [1], [2], [chunk: id]
 * - Links: [label](url)
 */
function renderInlineContent(text: string): React.ReactNode[] {
  // Regex splitting inline elements while capturing groups:
  // 1: bold (**...** or __...__)
  // 2: inline code (`...`)
  // 3: citation ([1] or [chunk: ...])
  // 4: markdown link ([...](...))
  // 5: italic (*...* or _..._)
  const inlineRegex = /(\*\*[^*]+\*\*|__[^_]+__|`[^`]+`|\[(?:chunk:\s*)?[a-zA-Z0-9_\-]+\]|\[[^\]]+\]\([^)]+\)|\*[^*]+\*|_[^_]+_)/g;

  const parts = text.split(inlineRegex);

  return parts.map((part, index) => {
    if (!part) return null;

    // Bold: **text** or __text__
    if ((part.startsWith('**') && part.endsWith('**')) || (part.startsWith('__') && part.endsWith('__'))) {
      const inner = part.slice(2, -2);
      return <strong key={index} style={{ fontWeight: 600, color: 'inherit' }}>{inner}</strong>;
    }

    // Inline Code: `code`
    if (part.startsWith('`') && part.endsWith('`')) {
      const inner = part.slice(1, -1);
      return (
        <code
          key={index}
          style={{
            fontFamily: 'var(--font-mono, monospace)',
            fontSize: '0.88em',
            padding: '2px 5px',
            borderRadius: '4px',
            background: 'rgba(0, 0, 0, 0.05)',
            color: 'var(--color-ink, #0f172a)',
            border: '1px solid rgba(0, 0, 0, 0.08)',
          }}
        >
          {inner}
        </code>
      );
    }

    // Citation badge: [1], [2], [chunk: abc]
    const citeMatch = part.match(/^\[((?:chunk:\s*)?[a-zA-Z0-9_\-]+)\]$/);
    if (citeMatch) {
      const label = citeMatch[1];
      return (
        <span
          key={index}
          title={`Source Citation: ${label}`}
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '10.5px',
            fontWeight: 600,
            padding: '1px 5px',
            margin: '0 2px',
            borderRadius: '4px',
            background: '#EFF6FF',
            color: '#1D4ED8',
            border: '1px solid #BFDBFE',
            verticalAlign: 'baseline',
            userSelect: 'none',
          }}
        >
          [{label}]
        </span>
      );
    }

    // Link: [label](url)
    const linkMatch = part.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
    if (linkMatch) {
      return (
        <a
          key={index}
          href={linkMatch[2]}
          target="_blank"
          rel="noopener noreferrer"
          style={{ color: 'var(--color-teal, #0284c7)', textDecoration: 'underline' }}
        >
          {linkMatch[1]}
        </a>
      );
    }

    // Italic: *text* or _text_
    if ((part.startsWith('*') && part.endsWith('*')) || (part.startsWith('_') && part.endsWith('_'))) {
      const inner = part.slice(1, -1);
      return <em key={index} style={{ fontStyle: 'italic' }}>{inner}</em>;
    }

    return <React.Fragment key={index}>{part}</React.Fragment>;
  });
}

/**
 * Main Markdown renderer component:
 * Accurately parses headers, lists, markdown tables, fenced code blocks,
 * blockquotes, paragraphs, and inline markdown styling.
 */
export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content, className, style }) => {
  if (!content) return null;

  const lines = content.split('\n');
  const elements: React.ReactNode[] = [];

  let i = 0;
  let keyIndex = 0;

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    // 1. Fenced Code Block
    if (trimmed.startsWith('```')) {
      const codeLines: string[] = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith('```')) {
        codeLines.push(lines[i]);
        i++;
      }
      i++; // skip closing ```
      elements.push(
        <pre
          key={keyIndex++}
          style={{
            background: '#0F172A',
            color: '#F8FAFC',
            padding: '12px 16px',
            borderRadius: '8px',
            overflowX: 'auto',
            fontSize: '12.5px',
            lineHeight: 1.5,
            margin: '10px 0',
            fontFamily: 'var(--font-mono, monospace)',
          }}
        >
          <code>{codeLines.join('\n')}</code>
        </pre>
      );
      continue;
    }

    // 2. Table detection (lines starting with | and contains |)
    if (trimmed.startsWith('|') && trimmed.endsWith('|') && i + 1 < lines.length && lines[i + 1].includes('---')) {
      const tableLines: string[] = [];
      while (i < lines.length && lines[i].trim().startsWith('|') && lines[i].trim().endsWith('|')) {
        tableLines.push(lines[i].trim());
        i++;
      }

      if (tableLines.length >= 2) {
        const headerRow = tableLines[0]
          .slice(1, -1)
          .split('|')
          .map((c) => c.trim());
        // Row 1 is divider (|---|---|), skip it
        const bodyRows = tableLines.slice(2).map((r) =>
          r
            .slice(1, -1)
            .split('|')
            .map((c) => c.trim())
        );

        elements.push(
          <div key={keyIndex++} style={{ overflowX: 'auto', margin: '12px 0' }}>
            <table
              style={{
                width: '100%',
                borderCollapse: 'collapse',
                fontSize: '13px',
                lineHeight: 1.5,
                border: '1px solid #E2E8F0',
                borderRadius: '6px',
                overflow: 'hidden',
              }}
            >
              <thead>
                <tr style={{ background: '#F8FAFC', borderBottom: '2px solid #CBD5E1' }}>
                  {headerRow.map((cell, cIdx) => (
                    <th
                      key={cIdx}
                      style={{
                        padding: '8px 12px',
                        textAlign: 'left',
                        fontWeight: 600,
                        color: '#1E293B',
                      }}
                    >
                      {renderInlineContent(cell)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {bodyRows.map((row, rIdx) => (
                  <tr
                    key={rIdx}
                    style={{
                      background: rIdx % 2 === 0 ? '#FFFFFF' : '#F8FAFC',
                      borderBottom: '1px solid #E2E8F0',
                    }}
                  >
                    {row.map((cell, cIdx) => (
                      <td
                        key={cIdx}
                        style={{
                          padding: '7px 12px',
                          color: '#334155',
                        }}
                      >
                        {renderInlineContent(cell)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
        continue;
      }
    }

    // 3. Headings
    if (trimmed.startsWith('# ')) {
      elements.push(
        <h1 key={keyIndex++} style={{ fontSize: '20px', fontWeight: 700, margin: '14px 0 8px', color: 'inherit' }}>
          {renderInlineContent(trimmed.slice(2))}
        </h1>
      );
      i++;
      continue;
    }
    if (trimmed.startsWith('## ')) {
      elements.push(
        <h2 key={keyIndex++} style={{ fontSize: '17px', fontWeight: 600, margin: '12px 0 6px', color: 'inherit' }}>
          {renderInlineContent(trimmed.slice(3))}
        </h2>
      );
      i++;
      continue;
    }
    if (trimmed.startsWith('### ')) {
      elements.push(
        <h3 key={keyIndex++} style={{ fontSize: '15px', fontWeight: 600, margin: '10px 0 4px', color: 'inherit' }}>
          {renderInlineContent(trimmed.slice(4))}
        </h3>
      );
      i++;
      continue;
    }

    // 4. Blockquotes
    if (trimmed.startsWith('> ')) {
      const quoteLines: string[] = [];
      while (i < lines.length && lines[i].trim().startsWith('>')) {
        quoteLines.push(lines[i].trim().replace(/^>\s?/, ''));
        i++;
      }
      elements.push(
        <blockquote
          key={keyIndex++}
          style={{
            margin: '8px 0',
            paddingLeft: '12px',
            borderLeft: '3px solid var(--color-teal, #0284c7)',
            color: 'var(--color-ink-muted, #64748b)',
            fontStyle: 'italic',
          }}
        >
          {quoteLines.map((l, qIdx) => (
            <div key={qIdx}>{renderInlineContent(l)}</div>
          ))}
        </blockquote>
      );
      continue;
    }

    // 5. Unordered Lists (- , * , + )
    if (/^[-*+]\s+/.test(trimmed)) {
      const items: string[] = [];
      while (i < lines.length && /^[-*+]\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^[-*+]\s+/, ''));
        i++;
      }
      elements.push(
        <ul
          key={keyIndex++}
          style={{
            margin: '6px 0 8px 18px',
            padding: 0,
            listStyleType: 'disc',
            display: 'flex',
            flexDirection: 'column',
            gap: '4px',
          }}
        >
          {items.map((it, itemIdx) => (
            <li key={itemIdx} style={{ lineHeight: 1.6 }}>
              {renderInlineContent(it)}
            </li>
          ))}
        </ul>
      );
      continue;
    }

    // 6. Ordered Lists (1. , 2. )
    if (/^\d+\.\s+/.test(trimmed)) {
      const items: string[] = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^\d+\.\s+/, ''));
        i++;
      }
      elements.push(
        <ol
          key={keyIndex++}
          style={{
            margin: '6px 0 8px 18px',
            padding: 0,
            listStyleType: 'decimal',
            display: 'flex',
            flexDirection: 'column',
            gap: '4px',
          }}
        >
          {items.map((it, itemIdx) => (
            <li key={itemIdx} style={{ lineHeight: 1.6 }}>
              {renderInlineContent(it)}
            </li>
          ))}
        </ol>
      );
      continue;
    }

    // 7. Regular paragraph / line
    if (trimmed === '') {
      i++;
      continue;
    }

    const paraLines: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() !== '' &&
      !lines[i].trim().startsWith('```') &&
      !lines[i].trim().startsWith('|') &&
      !lines[i].trim().startsWith('#') &&
      !lines[i].trim().startsWith('>') &&
      !/^[-*+]\s+/.test(lines[i].trim()) &&
      !/^\d+\.\s+/.test(lines[i].trim())
    ) {
      paraLines.push(lines[i]);
      i++;
    }

    elements.push(
      <p key={keyIndex++} style={{ margin: '6px 0', lineHeight: 1.65 }}>
        {paraLines.map((pLine, pIdx) => (
          <React.Fragment key={pIdx}>
            {pIdx > 0 && <br />}
            {renderInlineContent(pLine)}
          </React.Fragment>
        ))}
      </p>
    );
  }

  return (
    <div className={className} style={{ width: '100%', wordBreak: 'break-word', ...style }}>
      {elements}
    </div>
  );
};
