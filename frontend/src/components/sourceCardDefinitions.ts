import type { SourceItem, RetrievedChunk } from '../api/chat.ts';

/**
 * Validates whether a given string is a safe HTTP/HTTPS URL.
 * Strictly prevents javascript:, data:, and file: schemes.
 */
export function isValidWebUrl(url?: string | null): boolean {
  if (!url || typeof url !== 'string') return false;
  const trimmed = url.trim();
  try {
    const parsed = new URL(trimmed);
    return parsed.protocol === 'http:' || parsed.protocol === 'https:';
  } catch {
    return false;
  }
}

/**
 * Derives a clean, readable hostname from a valid HTTP/HTTPS URL.
 * Strips leading 'www.' for a cleaner display.
 */
export function extractHostname(url?: string | null): string | null {
  if (!isValidWebUrl(url)) return null;
  try {
    const parsed = new URL(url!.trim());
    return parsed.hostname.replace(/^www\./i, '');
  } catch {
    return null;
  }
}

/**
 * Deduplicates source items by URL, chunk ID, or filename+page combination.
 * Preserves initial ordering and merges excerpt if missing from earlier occurrence.
 */
export function deduplicateSources(sources: SourceItem[]): SourceItem[] {
  const seenMap = new Map<string, SourceItem>();
  const deduplicated: SourceItem[] = [];

  for (let idx = 0; idx < sources.length; idx++) {
    const s = sources[idx];
    const source1BasedNum = idx + 1;

    let key = '';
    if (s.url && isValidWebUrl(s.url)) {
      key = `url:${s.url.trim().toLowerCase()}`;
    } else if (s.chunk_id) {
      key = `chunk:${s.chunk_id.trim()}`;
    } else {
      const file = (s.file_name || s.title || '').trim().toLowerCase();
      const page = s.page_number ?? '';
      key = `file:${file}_p:${page}`;
    }

    const existing = seenMap.get(key);
    if (!existing) {
      const item: SourceItem = {
        ...s,
        indices: [source1BasedNum],
      };
      seenMap.set(key, item);
      deduplicated.push(item);
    } else {
      if (!existing.indices) existing.indices = [];
      if (!existing.indices.includes(source1BasedNum)) {
        existing.indices.push(source1BasedNum);
      }
      if (!existing.excerpt && s.excerpt) {
        existing.excerpt = s.excerpt;
      }
    }
  }

  return deduplicated;
}

/**
 * Normalizes raw sources, citations strings, or retrieved chunks into a uniform SourceItem list.
 */
export function normalizeSources(
  sources?: SourceItem[] | null,
  citations?: string[] | null,
  chunks?: RetrievedChunk[] | null
): SourceItem[] {
  const items: SourceItem[] = [];

  // 1. If explicit sources list is provided
  if (sources && Array.isArray(sources) && sources.length > 0) {
    for (const s of sources) {
      const isWeb =
        s.type === 'web_research' ||
        s.type === 'web' ||
        Boolean(s.url && isValidWebUrl(s.url)) ||
        Boolean(s.chunk_id && s.chunk_id.startsWith('web_'));

      const isSharedKn =
        s.type === 'shared_knowledge' ||
        Boolean(s.chunk_id && s.chunk_id.startsWith('shared_kn_'));

      const isThread = s.type === 'thread';
      const isMemory = s.type === 'personal_memory';

      let resolvedType: string = 'document';
      if (isWeb) resolvedType = 'web_research';
      else if (isSharedKn) resolvedType = 'shared_knowledge';
      else if (isThread) resolvedType = 'thread';
      else if (isMemory) resolvedType = 'personal_memory';

      items.push({
        ...s,
        type: resolvedType,
        url: isValidWebUrl(s.url) ? s.url : isValidWebUrl(s.document_id) ? s.document_id : undefined,
        file_name: s.file_name || undefined,
        page_number: typeof s.page_number === 'number' ? s.page_number : null,
        title: s.title || s.file_name || (isValidWebUrl(s.url) ? extractHostname(s.url) || 'Web Source' : 'Document'),
        excerpt: s.excerpt || s.text || undefined,
      });
    }
    return deduplicateSources(items);
  }

  // 2. If retrieved_chunks are provided
  if (chunks && Array.isArray(chunks) && chunks.length > 0) {
    for (const c of chunks) {
      const isWeb = c.chunk_id.startsWith('web_') || isValidWebUrl(c.document_id);
      const isSharedKn = c.chunk_id.startsWith('shared_kn_');

      items.push({
        type: isWeb ? 'web_research' : isSharedKn ? 'shared_knowledge' : 'document',
        chunk_id: c.chunk_id,
        document_id: c.document_id,
        department_id: c.department_id,
        file_name: c.file_name,
        page_number: typeof c.page_number === 'number' ? c.page_number : null,
        title: isWeb ? `Web: ${c.file_name || 'Source'}` : c.file_name,
        excerpt: c.text,
        url: isValidWebUrl(c.document_id) ? c.document_id : undefined,
      });
    }
    return deduplicateSources(items);
  }

  // 3. Fallback to citations strings if neither sources nor chunks are provided
  if (citations && Array.isArray(citations) && citations.length > 0) {
    for (const cite of citations) {
      if (!cite || typeof cite !== 'string') continue;
      const trimmed = cite.trim();
      if (!trimmed) continue;

      if (isValidWebUrl(trimmed)) {
        items.push({
          type: 'web_research',
          url: trimmed,
          title: extractHostname(trimmed) || trimmed,
        });
        continue;
      }

      // Check format like "01_document.pdf, page 4"
      const pageMatch = trimmed.match(/^(.+?),\s*(?:page|p\.?)\s*(\d+)$/i);
      if (pageMatch) {
        items.push({
          type: 'document',
          file_name: pageMatch[1].trim(),
          title: pageMatch[1].trim(),
          page_number: parseInt(pageMatch[2], 10),
        });
        continue;
      }

      // Standalone filename or chunk identifier
      items.push({
        type: 'document',
        file_name: trimmed,
        title: trimmed,
        page_number: null,
      });
    }
    return deduplicateSources(items);
  }

  return [];
}
