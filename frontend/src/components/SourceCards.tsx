import React, { useState, useEffect, useRef } from 'react';
import { SourceItem, RetrievedChunk } from '../api/chat';
import {
  isValidWebUrl,
  extractHostname,
  deduplicateSources,
  normalizeSources,
} from './sourceCardDefinitions';

export {
  isValidWebUrl,
  extractHostname,
  deduplicateSources,
  normalizeSources,
};

/* ═══════════════ INLINE ICONS (Consistent with Sidebar & Theme) ═══════════════ */

function IconDocument() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="16" y1="13" x2="8" y2="13" />
      <line x1="16" y1="17" x2="8" y2="17" />
    </svg>
  );
}

function IconGlobe() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="12" cy="12" r="10" />
      <line x1="2" y1="12" x2="22" y2="12" />
      <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
    </svg>
  );
}

function IconExternalLink() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
      <polyline points="15 3 21 3 21 9" />
      <line x1="10" y1="14" x2="21" y2="3" />
    </svg>
  );
}

function IconChevron({ expanded }: { expanded: boolean }) {
  return (
    <svg
      width="12"
      height="12"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      style={{
        transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)',
        transition: 'transform 0.15s ease',
      }}
    >
      <polyline points="6 9 12 15 18 9" />
    </svg>
  );
}

/* ═══════════════ INDIVIDUAL SOURCE CARD COMPONENT ═══════════════ */

interface SourceCardProps {
  source: SourceItem;
  index: number;
  isFocused: boolean;
  onClearFocus?: () => void;
  onInspectChunk?: (documentId: string, chunkId: string, fileName?: string) => void;
}

export function SingleSourceCard({ source, index, isFocused, onClearFocus, onInspectChunk }: SourceCardProps) {
  const cardRef = useRef<HTMLDivElement>(null);
  const [excerptExpanded, setExcerptExpanded] = useState(false);

  const isWeb = source.type === 'web_research' || Boolean(source.url && isValidWebUrl(source.url));
  const isShared = source.type === 'shared_knowledge';
  const isThread = source.type === 'thread';
  const isMemory = source.type === 'personal_memory';

  const hostname = isWeb ? extractHostname(source.url) : null;
  const hasValidLink = isWeb && isValidWebUrl(source.url);

  // Scroll into view when focused by inline citation
  useEffect(() => {
    if (isFocused && cardRef.current) {
      cardRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      const timer = setTimeout(() => {
        if (onClearFocus) onClearFocus();
      }, 2500);
      return () => clearTimeout(timer);
    }
  }, [isFocused, onClearFocus]);

  // Clean title
  let displayTitle = source.title || source.file_name || (hostname ? `Web: ${hostname}` : 'Source');
  if (isWeb && displayTitle.startsWith('Web: ')) {
    displayTitle = displayTitle.slice(5).trim();
  }

  // Type label
  let typeLabel = 'Internal Document';
  if (isWeb) typeLabel = 'Web Source';
  else if (isShared) typeLabel = 'Shared Knowledge';
  else if (isThread) typeLabel = 'Workspace Thread';
  else if (isMemory) typeLabel = 'Personal Preference';

  const hasExcerpt = Boolean(source.excerpt && source.excerpt.trim().length > 0);
  const excerptText = source.excerpt ? source.excerpt.trim() : '';
  const isLongExcerpt = excerptText.length > 200;

  return (
    <div
      ref={cardRef}
      id={`source-card-${index + 1}`}
      tabIndex={-1}
      style={{
        background: isFocused ? 'var(--color-teal-light)' : 'var(--color-surface)',
        border: isFocused ? '1px solid var(--color-teal)' : '1px solid var(--color-border)',
        boxShadow: isFocused ? '0 0 0 2px rgba(61, 122, 120, 0.25)' : 'none',
        borderRadius: '6px',
        padding: '10px 12px',
        display: 'flex',
        flexDirection: 'column',
        gap: '6px',
        transition: 'all 0.2s ease',
        fontFamily: 'var(--font-sans)',
      }}
    >
      {/* Top Header Row: Icon, Source Type, Citation Badge */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span
            style={{
              color: isWeb ? 'var(--color-teal)' : 'var(--color-ink-muted)',
              display: 'flex',
              alignItems: 'center',
            }}
          >
            {isWeb ? <IconGlobe /> : <IconDocument />}
          </span>
          <span
            style={{
              fontSize: '10px',
              fontFamily: 'var(--font-mono)',
              textTransform: 'uppercase',
              letterSpacing: '0.06em',
              fontWeight: 600,
              color: 'var(--color-ink-muted)',
            }}
          >
            {typeLabel}
          </span>
        </div>

        {/* Citation identifier badge */}
        <span
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '10px',
            fontWeight: 600,
            padding: '1px 6px',
            borderRadius: '4px',
            background: isFocused ? 'var(--color-teal)' : 'var(--color-panel)',
            color: isFocused ? '#ffffff' : 'var(--color-ink)',
            border: '1px solid var(--color-border)',
            userSelect: 'none',
          }}
          title={`Citation reference [${index + 1}]`}
        >
          [{index + 1}]
        </span>
      </div>

      {/* Main Title */}
      <div
        style={{
          fontSize: '13px',
          fontWeight: 600,
          color: 'var(--color-ink)',
          lineHeight: 1.4,
          wordBreak: 'break-word',
        }}
      >
        {displayTitle}
      </div>

      {/* Secondary Line: Hostname, Page number, Open Source link */}
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          alignItems: 'center',
          gap: '8px',
          fontSize: '11px',
          color: 'var(--color-ink-muted)',
          lineHeight: 1.4,
        }}
      >
        {/* Hostname for web sources */}
        {hostname && (
          <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-ink-muted)' }}>
            {hostname}
          </span>
        )}

        {/* Page number for internal documents */}
        {typeof source.page_number === 'number' && (
          <span
            style={{
              fontFamily: 'var(--font-mono)',
              background: 'var(--color-ground)',
              border: '1px solid var(--color-border)',
              padding: '1px 5px',
              borderRadius: '3px',
              fontSize: '10.5px',
              color: 'var(--color-ink)',
            }}
          >
            Page {source.page_number}
          </span>
        )}

        {/* Department / Workspace metadata if present */}
        {source.workspace_name && (
          <span style={{ fontSize: '10.5px' }}>
            {source.workspace_name}
          </span>
        )}

        {/* Chunk History & Diff inspection button for document chunks */}
        {source.document_id && source.chunk_id && onInspectChunk && (
          <button
            type="button"
            onClick={() => onInspectChunk(source.document_id!, source.chunk_id!, source.file_name || source.title)}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '4px',
              color: 'var(--color-teal, #0F766E)',
              background: 'transparent',
              border: 'none',
              padding: '2px 4px',
              cursor: 'pointer',
              fontSize: '11px',
              fontWeight: 600,
              marginLeft: hasValidLink ? '0' : 'auto',
            }}
            title="Inspect chunk version history, diffs, and curation lifecycle"
          >
            <span>📜 History & Diff</span>
          </button>
        )}

        {/* Valid HTTP/HTTPS external link */}
        {hasValidLink && (
          <a
            href={source.url!}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '4px',
              color: 'var(--color-teal)',
              textDecoration: 'none',
              fontWeight: 500,
              fontSize: '11px',
              marginLeft: 'auto',
              transition: 'opacity 0.15s ease',
            }}
            onMouseEnter={(e) => (e.currentTarget.style.textDecoration = 'underline')}
            onMouseLeave={(e) => (e.currentTarget.style.textDecoration = 'none')}
            title={`Open source: ${source.url}`}
          >
            <span>Open source</span>
            <IconExternalLink />
          </a>
        )}
      </div>

      {/* Evidence Excerpt region (only when actual text is available) */}
      {hasExcerpt && (
        <div
          style={{
            marginTop: '4px',
            background: 'var(--color-ground)',
            borderLeft: '2px solid var(--color-teal)',
            borderRadius: '2px 4px 4px 2px',
            padding: '6px 10px',
            fontSize: '11.5px',
            lineHeight: 1.5,
            color: 'var(--color-ink-muted)',
          }}
        >
          <div
            style={{
              wordBreak: 'break-word',
              display: excerptExpanded || !isLongExcerpt ? 'block' : '-webkit-box',
              WebkitLineClamp: excerptExpanded || !isLongExcerpt ? 'unset' : 3,
              WebkitBoxOrient: 'vertical',
              overflow: excerptExpanded || !isLongExcerpt ? 'visible' : 'hidden',
            }}
          >
            "{excerptText}"
          </div>
          {isLongExcerpt && (
            <button
              type="button"
              onClick={() => setExcerptExpanded(!excerptExpanded)}
              style={{
                background: 'transparent',
                border: 'none',
                padding: '2px 0 0 0',
                color: 'var(--color-teal)',
                fontSize: '10.5px',
                cursor: 'pointer',
                fontWeight: 600,
              }}
            >
              {excerptExpanded ? 'Show less' : 'Show full excerpt'}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

/* ═══════════════ MAIN SOURCES SECTION COMPONENT ═══════════════ */

export interface SourceCardsProps {
  sources?: SourceItem[] | null;
  citations?: string[] | null;
  chunks?: RetrievedChunk[] | null;
  focusedCitation?: string | null;
  onClearFocusedCitation?: () => void;
  onInspectChunk?: (documentId: string, chunkId: string, fileName?: string) => void;
  initiallyExpanded?: boolean;
  className?: string;
  style?: React.CSSProperties;
}

export const SourceCards: React.FC<SourceCardsProps> = ({
  sources,
  citations,
  chunks,
  focusedCitation,
  onClearFocusedCitation,
  onInspectChunk,
  initiallyExpanded = false,
  className,
  style,
}) => {
  const normalized = normalizeSources(sources, citations, chunks);
  const [isExpanded, setIsExpanded] = useState<boolean>(initiallyExpanded);

  // If an inline citation is clicked, auto-expand the sources section
  useEffect(() => {
    if (focusedCitation && !isExpanded) {
      setIsExpanded(true);
    }
  }, [focusedCitation, isExpanded]);

  // Do not render empty sources section if no citations or sources exist
  if (!normalized || normalized.length === 0) {
    return null;
  }

  // Count internal vs web sources for informative summary
  const webCount = normalized.filter((s) => s.type === 'web_research' || isValidWebUrl(s.url)).length;
  const internalCount = normalized.length - webCount;

  let summaryParts: string[] = [];
  if (internalCount > 0) summaryParts.push(`${internalCount} document${internalCount !== 1 ? 's' : ''}`);
  if (webCount > 0) summaryParts.push(`${webCount} web`);
  const summaryText = summaryParts.join(' • ');

  const checkIsFocused = (source: SourceItem, idx: number): boolean => {
    if (!focusedCitation) return false;
    const cleanFocus = focusedCitation.trim().toLowerCase();

    // 1. Matches numeric index: "1" matches index 0
    if (cleanFocus === String(idx + 1)) return true;

    // 2. Matches source chunk index list (e.g. citation "5" matches card covering chunk 5)
    if (source.indices && Array.isArray(source.indices)) {
      const numFocus = parseInt(cleanFocus, 10);
      if (!isNaN(numFocus) && source.indices.includes(numFocus)) return true;
    }

    // 3. Matches [chunk: id] or raw chunk_id
    if (source.chunk_id) {
      const cleanChunk = source.chunk_id.trim().toLowerCase();
      if (cleanFocus === cleanChunk) return true;
      if (cleanFocus === `chunk: ${cleanChunk}` || cleanFocus === `chunk:${cleanChunk}`) return true;
      if (cleanFocus.replace(/^chunk:\s*/i, '') === cleanChunk) return true;
    }

    // 4. Matches filename
    if (source.file_name && cleanFocus.includes(source.file_name.toLowerCase())) {
      return true;
    }

    return false;
  };

  return (
    <div
      className={className}
      style={{
        marginTop: '12px',
        borderTop: '1px solid var(--color-border)',
        paddingTop: '8px',
        ...style,
      }}
    >
      {/* Collapsible Trigger Button */}
      <button
        type="button"
        onClick={() => setIsExpanded(!isExpanded)}
        aria-expanded={isExpanded}
        aria-controls="sources-cards-list"
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '6px',
          background: isExpanded ? 'var(--color-panel)' : 'transparent',
          border: '1px solid var(--color-border)',
          borderRadius: '4px',
          padding: '4px 8px',
          cursor: 'pointer',
          color: 'var(--color-ink)',
          fontFamily: 'var(--font-sans)',
          fontSize: '11px',
          fontWeight: 600,
          transition: 'all 0.15s ease',
          userSelect: 'none',
        }}
        onMouseEnter={(e) => {
          if (!isExpanded) e.currentTarget.style.background = 'var(--color-panel)';
        }}
        onMouseLeave={(e) => {
          if (!isExpanded) e.currentTarget.style.background = 'transparent';
        }}
      >
        <IconChevron expanded={isExpanded} />
        <span>Sources ({normalized.length})</span>
        {summaryText && (
          <span
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '10px',
              color: 'var(--color-ink-muted)',
              fontWeight: 400,
              marginLeft: '2px',
            }}
          >
            [{summaryText}]
          </span>
        )}
      </button>

      {/* Expanded Sources Grid / List */}
      {isExpanded && (
        <div
          id="sources-cards-list"
          style={{
            marginTop: '8px',
            display: 'flex',
            flexDirection: 'column',
            gap: '8px',
          }}
        >
          {normalized.map((source, idx) => (
            <SingleSourceCard
              key={source.chunk_id || source.url || `${source.file_name}-${idx}`}
              source={source}
              index={idx}
              isFocused={checkIsFocused(source, idx)}
              onClearFocus={onClearFocusedCitation}
              onInspectChunk={onInspectChunk}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default SourceCards;
