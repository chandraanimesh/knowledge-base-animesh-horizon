/**
 * Domain definitions and pure helper functions for Milestone 7A.2:
 * Chunk History UI, Diff Viewer, and Curation Lifecycle.
 */

export interface DiffLine {
  type: 'header' | 'addition' | 'deletion' | 'context' | 'meta';
  content: string;
  oldLineNumber?: number | null;
  newLineNumber?: number | null;
}

export interface CurationPermissionCheck {
  allowed: boolean;
  reason?: string;
  canEdit: boolean;
  canRollback: boolean;
  isManaged: boolean;
}

/**
 * Parses a standard unified diff string into structured lines with line numbering.
 */
export function parseUnifiedDiff(diffPatch: string): DiffLine[] {
  if (!diffPatch || !diffPatch.trim()) {
    return [];
  }

  const rawLines = diffPatch.split('\n');
  const result: DiffLine[] = [];

  let oldLine = 1;
  let newLine = 1;

  for (const line of rawLines) {
    if (line.startsWith('---') || line.startsWith('+++')) {
      result.push({
        type: 'meta',
        content: line,
      });
    } else if (line.startsWith('@@')) {
      // Hunk header: e.g. @@ -1,4 +1,5 @@
      const match = line.match(/@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/);
      if (match) {
        oldLine = parseInt(match[1], 10);
        newLine = parseInt(match[2], 10);
      }
      result.push({
        type: 'header',
        content: line,
      });
    } else if (line.startsWith('+')) {
      result.push({
        type: 'addition',
        content: line.slice(1),
        newLineNumber: newLine,
      });
      newLine++;
    } else if (line.startsWith('-')) {
      result.push({
        type: 'deletion',
        content: line.slice(1),
        oldLineNumber: oldLine,
      });
      oldLine++;
    } else {
      // Context or empty line
      const content = line.startsWith(' ') ? line.slice(1) : line;
      result.push({
        type: 'context',
        content,
        oldLineNumber: oldLine,
        newLineNumber: newLine,
      });
      oldLine++;
      newLine++;
    }
  }

  return result;
}

/**
 * Evaluates whether a caller is authorized to curate (edit / rollback) a chunk.
 * Preserves fail-closed behavior for viewers, unauthorized users, and managed CRM chunks.
 */
export function checkCurationPermissions(
  userRole?: string | null,
  workspaceRoleOrManaged?: string | boolean | null,
  isManagedExternal = false
): CurationPermissionCheck {
  let isManaged = isManagedExternal;
  let wsRole: string | null = null;
  if (typeof workspaceRoleOrManaged === 'boolean') {
    isManaged = workspaceRoleOrManaged;
  } else if (typeof workspaceRoleOrManaged === 'string') {
    wsRole = workspaceRoleOrManaged;
  }

  const normUserRole = (userRole || 'USER').toUpperCase();
  const normWsRole = (wsRole || '').toUpperCase();

  // Rule 1: Managed CRM chunks are always read-only
  if (isManaged) {
    return {
      allowed: false,
      reason: 'This chunk is managed by Twenty CRM sync. Manual curation is disabled to prevent sync drift.',
      canEdit: false,
      canRollback: false,
      isManaged: true,
    };
  }

  // Rule 2: ADMIN, OWNER, or EDITOR has curation access
  if (
    normUserRole === 'ADMIN' ||
    normUserRole === 'OWNER' ||
    normUserRole === 'EDITOR' ||
    normWsRole === 'OWNER' ||
    normWsRole === 'EDITOR'
  ) {
    return {
      allowed: true,
      canEdit: true,
      canRollback: true,
      isManaged: false,
    };
  }

  if (normUserRole === 'VIEWER' || normWsRole === 'VIEWER') {
    return {
      allowed: false,
      reason: 'Viewer role has read-only access. Curation requires Editor or Owner privileges.',
      canEdit: false,
      canRollback: false,
      isManaged: false,
    };
  }

  // Default deny
  return {
    allowed: false,
    reason: 'Insufficient permissions to modify knowledge chunks.',
    canEdit: false,
    canRollback: false,
    isManaged: false,
  };
}

/**
 * Truncates a SHA-256 hash for compact badge display.
 */
export function formatChunkHash(hash?: string | null, length = 10): string {
  if (!hash) return '—';
  if (hash.length <= length) return hash;
  return `${hash.slice(0, length)}…`;
}

/**
 * Evaluates whether a chunk version can be rolled back to.
 */
export function canRollbackToVersion(
  targetVersion: number,
  currentVersion: number,
  canRollback: boolean
): boolean {
  if (!canRollback) return false;
  if (!targetVersion || targetVersion <= 0) return false;
  return targetVersion !== currentVersion;
}

/**
 * Formats ISO timestamps into human-readable locale strings.
 */
export function formatAuditTimestamp(isoDate?: string | null): string {
  if (!isoDate) return '—';
  try {
    const d = new Date(isoDate);
    if (isNaN(d.getTime())) return isoDate;
    return d.toLocaleString(undefined, {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return isoDate;
  }
}

/**
 * Checks whether an API error represents an optimistic locking HTTP 409 conflict.
 */
export function isOptimisticConflictError(error: any): boolean {
  if (!error) return false;
  if (typeof error === 'object' && (error.status === 409 || error.statusCode === 409)) return true;
  const raw = typeof error === 'string' ? error : (error.message || '');
  const msg = String(raw).toLowerCase();
  return (
    msg.includes('409') ||
    msg.includes('conflict') ||
    msg.includes('base_version') ||
    msg.includes('modified by another')
  );
}
