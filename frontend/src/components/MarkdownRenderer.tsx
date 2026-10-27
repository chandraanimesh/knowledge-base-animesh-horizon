import React from 'react';
import { ArtifactMetadata } from '../api/chat';
import { downloadArtifactFile } from './ArtifactList';

interface MarkdownRendererProps {
  content: string;
  className?: string;
  style?: React.CSSProperties;
  artifacts?: ArtifactMetadata[];
  onDownloadArtifact?: (artifact: ArtifactMetadata) => void;
  onCitationClick?: (citationLabel: string) => void;
}

function lookupArtifact(
  rawKey: string,
  artifactMap?: Map<string, ArtifactMetadata>
): ArtifactMetadata | undefined {
  if (!artifactMap || !rawKey) return undefined;
  const clean = rawKey.trim().toLowerCase().replace(/^[`'"]+|[`'"]+$/g, '');
  if (!clean) return undefined;

  // Direct match
  if (artifactMap.has(clean)) return artifactMap.get(clean);

  // Normalized key (strip punctuation / separators)
  const noSeparators = clean.replace(/[_\- ]+/g, '');
  if (artifactMap.has(noSeparators)) return artifactMap.get(noSeparators);

  const underscoreNorm = clean.replace(/[_\- ]+/g, '_');
  if (artifactMap.has(underscoreNorm)) return artifactMap.get(underscoreNorm);

  // Check if string ends with or contains an extension matching format
  for (const ext of ['pdf', 'json', 'csv', 'docx']) {
    if (clean.includes(`.${ext}`) || clean.endsWith(ext)) {
      const art = artifactMap.get(ext);
      if (art) return art;
    }
  }

  return undefined;
}

/**
 * Parses inline markdown:
 * - Bold: **text** or __text__
 * - Italic: *text* or _text_
 * - Inline code: `code` (converts to clickable download if matching artifact)
 * - Standalone filename: File.pdf / file.json (converts to clickable download if matching artifact)
 * - Citations: [1], [2], [chunk: id]
 * - Links: [label](url)
 */
function renderInlineContent(
  text: string,
  artifactMap?: Map<string, ArtifactMetadata>,
  onDownloadArtifact?: (artifact: ArtifactMetadata) => void,
  onCitationClick?: (citationLabel: string) => void
): React.ReactNode[] {
  // Regex splitting inline elements while capturing groups:
  // 1: bold (**...** or __...__)
  // 2: inline code (`...`)
  // 3: filename (e.g. Sales_Discount_Rules.pdf)
  // 4: citation ([1] or [chunk: ...] or 【1】)
  // 5: markdown link ([...](...))
  // 6: italic (*...* or _..._)
  const inlineRegex = /(\*\*[^*]+\*\*|__[^_]+__|`[^`]+`|\b[a-zA-Z0-9_\-]+\.(?:pdf|json|csv|docx)\b|(?:\[|【|［|〔)(?:chunk:\s*)?[a-zA-Z0-9_\-]+(?:\]|】|］|〕)|\[[^\]]+\]\([^)]+\)|\*[^*]+\*|_[^_]+_)/gi;

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
      const matchArt = lookupArtifact(inner, artifactMap);
      if (matchArt) {
        return (
          <button
            key={index}
            type="button"
            onClick={() => onDownloadArtifact ? onDownloadArtifact(matchArt) : downloadArtifactFile(matchArt)}
            title={`Download ${matchArt.filename}`}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '4px',
              fontFamily: 'var(--font-mono, monospace)',
              fontSize: '0.85em',
              padding: '2px 8px',
              borderRadius: '4px',
              background: '#EFF6FF',
              color: '#1D4ED8',
              border: '1px solid #BFDBFE',
              cursor: 'pointer',
              fontWeight: 600,
              verticalAlign: 'baseline',
            }}
          >
            <span>⬇️</span>
            <span>{matchArt.filename}</span>
          </button>
        );
      }
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

    // Standalone filename token matching an artifact (e.g. Sales_Discount_Approval_Rules.pdf)
    const fileMatch = part.match(/^([a-zA-Z0-9_\-]+\.(?:pdf|json|csv|docx))$/i);
    if (fileMatch) {
      const matchArt = lookupArtifact(fileMatch[1], artifactMap);
      if (matchArt) {
        return (
          <button
            key={index}
            type="button"
            onClick={() => onDownloadArtifact ? onDownloadArtifact(matchArt) : downloadArtifactFile(matchArt)}
            title={`Download ${matchArt.filename}`}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '4px',
              fontFamily: 'var(--font-mono, monospace)',
              fontSize: '0.85em',
              padding: '2px 8px',
              borderRadius: '4px',
              background: '#EFF6FF',
              color: '#1D4ED8',
              border: '1px solid #BFDBFE',
              cursor: 'pointer',
              fontWeight: 600,
              verticalAlign: 'baseline',
            }}
          >
            <span>⬇️</span>
            <span>{matchArt.filename}</span>
          </button>
        );
      }
    }

    // Citation badge: [1], [2], [chunk: abc], 【1】, 【2】, etc.
    const citeMatch = part.match(/^(?:\[|【|［|〔)((?:chunk:\s*)?[a-zA-Z0-9_\-]+)(?:\]|】|］|〕)$/);
    if (citeMatch) {
      const label = citeMatch[1];
      const isInteractive = Boolean(onCitationClick);
      return (
        <button
          key={index}
          type="button"
          onClick={() => onCitationClick && onCitationClick(label)}
          title={`View source citation: [${label}]`}
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '10.5px',
            fontFamily: 'var(--font-mono, monospace)',
            fontWeight: 600,
            padding: '1px 5px',
            margin: '0 2px',
            borderRadius: '4px',
            background: 'var(--color-teal-light, #e4f0ef)',
            color: 'var(--color-teal-dim, #2d5a58)',
            border: '1px solid var(--color-teal, #3d7a78)',
            verticalAlign: 'baseline',
            cursor: isInteractive ? 'pointer' : 'default',
            userSelect: 'none',
            lineHeight: 'normal',
            transition: 'all 0.15s ease',
          }}
          onMouseEnter={(e) => {
            if (isInteractive) {
              e.currentTarget.style.background = 'var(--color-teal, #3d7a78)';
              e.currentTarget.style.color = '#ffffff';
            }
          }}
          onMouseLeave={(e) => {
            if (isInteractive) {
              e.currentTarget.style.background = 'var(--color-teal-light, #e4f0ef)';
              e.currentTarget.style.color = 'var(--color-teal-dim, #2d5a58)';
            }
          }}
        >
          [{label}]
        </button>
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
export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({
  content,
  className,
  style,
  artifacts,
  onDownloadArtifact,
  onCitationClick,
}) => {
  if (!content) return null;

  const artifactMap = React.useMemo(() => {
    if (!artifacts || artifacts.length === 0) return undefined;
    const map = new Map<string, ArtifactMetadata>();
    for (const art of artifacts) {
      if (art.filename) {
        const lower = art.filename.toLowerCase();
        map.set(lower, art);
        map.set(art.filename.replace(/\.[^/.]+$/, '').toLowerCase(), art);
        map.set(lower.replace(/[_\- ]+/g, '_'), art);
        map.set(lower.replace(/[_\- ]+/g, ' '), art);
        map.set(lower.replace(/[_\- ]+/g, ''), art);
      }
      if (art.format) {
        const fmt = String(art.format).toLowerCase();
        map.set(fmt, art);
        map.set(`.${fmt}`, art);
      }
    }
    return map;
  }, [artifacts]);

  const renderInline = (t: string) =>
    renderInlineContent(t, artifactMap, onDownloadArtifact, onCitationClick);

  const lines = content.split('\n');
  const elements: React.ReactNode[] = [];

  let i = 0;
  let keyIndex = 0;

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    // 1. Fenced Code Block
    if (trimmed.startsWith('```')) {
      const lang = trimmed.slice(3).trim() || 'code';
      const codeLines: string[] = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith('```')) {
        codeLines.push(lines[i]);
        i++;
      }
      i++; // skip closing ```
      const fullCode = codeLines.join('\n');
      elements.push(
        <div
          key={keyIndex++}
          style={{
            margin: '12px 0',
            borderRadius: '8px',
            overflow: 'hidden',
            border: '1px solid #334155',
            background: '#0F172A',
          }}
        >
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: '6px 12px',
              background: '#1E293B',
              fontSize: '11px',
              color: '#94A3B8',
              fontFamily: 'var(--font-mono, monospace)',
            }}
          >
            <span>{lang.toUpperCase()}</span>
            <button
              type="button"
              onClick={() => navigator.clipboard?.writeText(fullCode)}
              style={{
                background: 'transparent',
                border: '1px solid #475569',
                borderRadius: '4px',
                color: '#CBD5E1',
                padding: '2px 8px',
                cursor: 'pointer',
                fontSize: '11px',
              }}
            >
              Copy
            </button>
          </div>
          <pre
            style={{
              color: '#F8FAFC',
              padding: '12px 16px',
              overflowX: 'auto',
              fontSize: '12.5px',
              lineHeight: 1.5,
              margin: 0,
              fontFamily: 'var(--font-mono, monospace)',
            }}
          >
            <code>{fullCode}</code>
          </pre>
        </div>
      );
      continue;
    }

    // 2. Table detection (lines starting with |)
    if (trimmed.startsWith('|')) {
      const tableLines: string[] = [];
      let j = i;
      while (j < lines.length && lines[j].trim().startsWith('|')) {
        tableLines.push(lines[j].trim());
        j++;
      }

      if (tableLines.length >= 2 && tableLines[1].includes('---')) {
        i = j; // Advance past all table rows
        const headerRow = tableLines[0]
          .replace(/^\|/, '')
          .replace(/\|$/, '')
          .split('|')
          .map((c) => c.trim());
        // Row 1 is divider (|---|---|), skip it
        const bodyRows = tableLines.slice(2).map((r) =>
          r
            .replace(/^\|/, '')
            .replace(/\|$/, '')
            .split('|')
            .map((c) => c.trim())
        );

        elements.push(
          <div
            key={keyIndex++}
            style={{
              overflowX: 'auto',
              WebkitOverflowScrolling: 'touch',
              margin: '14px 0',
              border: '1px solid #E2E8F0',
              borderRadius: '8px',
              boxShadow: '0 1px 3px rgba(0, 0, 0, 0.04)',
              background: '#FFFFFF',
              maxWidth: '100%',
            }}
          >
            <table
              style={{
                width: '100%',
                minWidth: Math.max(headerRow.length * 120, 520),
                borderCollapse: 'collapse',
                fontSize: '12.5px',
                lineHeight: 1.5,
              }}
            >
              <thead>
                <tr style={{ background: '#F8FAFC', borderBottom: '2px solid #E2E8F0' }}>
                  {headerRow.map((cell, cIdx) => (
                    <th
                      key={cIdx}
                      style={{
                        padding: '10px 14px',
                        textAlign: 'left',
                        fontWeight: 600,
                        color: '#1E293B',
                        fontSize: '12px',
                        letterSpacing: '0.02em',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {renderInline(cell)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {bodyRows.map((row, rIdx) => (
                  <tr
                    key={rIdx}
                    style={{
                      background: rIdx % 2 === 0 ? '#FFFFFF' : '#FBFCFD',
                      borderBottom: '1px solid #F1F5F9',
                    }}
                  >
                    {row.map((cell, cIdx) => (
                      <td
                        key={cIdx}
                        style={{
                          padding: '9px 14px',
                          color: '#334155',
                          wordBreak: 'break-word',
                          verticalAlign: 'top',
                        }}
                      >
                        {renderInline(cell)}
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
          {renderInline(trimmed.slice(2))}
        </h1>
      );
      i++;
      continue;
    }
    if (trimmed.startsWith('## ')) {
      elements.push(
        <h2 key={keyIndex++} style={{ fontSize: '17px', fontWeight: 600, margin: '12px 0 6px', color: 'inherit' }}>
          {renderInline(trimmed.slice(3))}
        </h2>
      );
      i++;
      continue;
    }
    if (trimmed.startsWith('### ')) {
      elements.push(
        <h3 key={keyIndex++} style={{ fontSize: '15px', fontWeight: 600, margin: '10px 0 4px', color: 'inherit' }}>
          {renderInline(trimmed.slice(4))}
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
            <div key={qIdx}>{renderInline(l)}</div>
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
          {items.map((it, itemIdx) => {
            const matchArt = lookupArtifact(it, artifactMap);
            if (matchArt) {
              return (
                <li key={itemIdx} style={{ lineHeight: 1.6, listStyleType: 'none', marginLeft: '-14px' }}>
                  <button
                    type="button"
                    onClick={() => onDownloadArtifact ? onDownloadArtifact(matchArt) : downloadArtifactFile(matchArt)}
                    title={`Download ${matchArt.filename}`}
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '6px',
                      padding: '4px 10px',
                      borderRadius: '6px',
                      background: '#F0F9FF',
                      border: '1px solid #BAE6FD',
                      color: '#0369A1',
                      cursor: 'pointer',
                      fontSize: '12px',
                      fontWeight: 600,
                      margin: '2px 0',
                    }}
                  >
                    <span>⬇️</span>
                    <span>{matchArt.filename}</span>
                    <span style={{ fontSize: '10.5px', color: '#0284C7', fontWeight: 500 }}>
                      ({matchArt.format.toUpperCase()} · {matchArt.size_bytes ? `${Math.round(matchArt.size_bytes / 1024)} KB` : ''})
                    </span>
                  </button>
                </li>
              );
            }
            return (
              <li key={itemIdx} style={{ lineHeight: 1.6 }}>
                {renderInline(it)}
              </li>
            );
          })}
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
              {renderInline(it)}
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

    const paraLines: string[] = [lines[i]];
    i++;
    while (
      i < lines.length &&
      lines[i].trim() !== '' &&
      !lines[i].trim().startsWith('```') &&
      !(lines[i].trim().startsWith('|') && i + 1 < lines.length && lines[i + 1].includes('---')) &&
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
            {renderInline(pLine)}
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
