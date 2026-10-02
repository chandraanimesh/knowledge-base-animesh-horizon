import React, { useState, useEffect, useCallback } from 'react';
import {
  WorkspaceItem,
  WorkspaceMemberItem,
  WorkspaceRole,
  workspaceApi,
} from '../api/workspaces';

interface WorkspaceMembersModalProps {
  workspace: WorkspaceItem;
  token: string;
  currentUserEmail?: string;
  isOwnerOrAdmin: boolean;
  onClose: () => void;
  onToast?: (type: 'success' | 'error', message: string) => void;
}

const ROLE_COLORS: Record<WorkspaceRole, { bg: string; color: string; border: string }> = {
  OWNER: { bg: '#e0f2f1', color: '#00695c', border: '#b2dfdb' },
  EDITOR: { bg: '#e3f2fd', color: '#1565c0', border: '#bbdefb' },
  VIEWER: { bg: '#f3f4f6', color: '#4b5563', border: '#e5e7eb' },
};

export default function WorkspaceMembersModal({
  workspace,
  token,
  currentUserEmail,
  isOwnerOrAdmin,
  onClose,
  onToast,
}: WorkspaceMembersModalProps) {
  const [members, setMembers] = useState<WorkspaceMemberItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Add member form state
  const [newUserId, setNewUserId] = useState('');
  const [newRole, setNewRole] = useState<WorkspaceRole>('VIEWER');
  const [adding, setAdding] = useState(false);
  const [updatingId, setUpdatingId] = useState<string | null>(null);

  const fetchMembers = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await workspaceApi.listWorkspaceMembers(token, workspace.id);
      setMembers(data);
    } catch (err: any) {
      const msg = err.message || 'Failed to load workspace members';
      setError(msg);
      onToast?.('error', msg);
    } finally {
      setLoading(false);
    }
  }, [token, workspace.id, onToast]);

  useEffect(() => {
    fetchMembers();
  }, [fetchMembers]);

  const handleAddMember = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newUserId.trim()) return;

    try {
      setAdding(true);
      await workspaceApi.addWorkspaceMember(token, workspace.id, {
        user_id: newUserId.trim(),
        role: newRole,
      });
      setNewUserId('');
      setNewRole('VIEWER');
      onToast?.('success', 'Member added successfully');
      await fetchMembers();
    } catch (err: any) {
      const msg = err.message || 'Failed to add member';
      onToast?.('error', msg);
    } finally {
      setAdding(false);
    }
  };

  const handleRoleChange = async (userId: string, role: WorkspaceRole) => {
    try {
      setUpdatingId(userId);
      await workspaceApi.updateWorkspaceMemberRole(token, workspace.id, userId, { role });
      onToast?.('success', 'Member role updated');
      await fetchMembers();
    } catch (err: any) {
      const msg = err.message || 'Failed to update member role';
      onToast?.('error', msg);
    } finally {
      setUpdatingId(null);
    }
  };

  const handleRemoveMember = async (userId: string) => {
    if (!confirm('Are you sure you want to remove this member from the workspace?')) return;

    try {
      setUpdatingId(userId);
      await workspaceApi.removeWorkspaceMember(token, workspace.id, userId);
      onToast?.('success', 'Member removed');
      await fetchMembers();
    } catch (err: any) {
      const msg = err.message || 'Failed to remove member';
      onToast?.('error', msg);
    } finally {
      setUpdatingId(null);
    }
  };

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(21, 29, 50, 0.5)',
        backdropFilter: 'blur(4px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 3100,
      }}
      onClick={onClose}
    >
      <div
        style={{
          background: 'var(--color-surface, #ffffff)',
          borderRadius: '16px',
          width: '100%',
          maxWidth: '620px',
          boxShadow: '0 20px 60px rgba(0,0,0,0.15)',
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
          maxHeight: '85vh',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            padding: '20px 24px',
            borderBottom: '1px solid var(--color-border, #e5e7eb)',
          }}
        >
          <div>
            <h3 style={{ margin: 0, fontSize: '18px', fontWeight: 600, color: 'var(--color-ink, #111827)' }}>
              Workspace Members
            </h3>
            <p style={{ margin: '4px 0 0', fontSize: '13px', color: 'var(--color-ink-muted, #6b7280)' }}>
              <strong>{workspace.name}</strong> ({workspace.code}) · Department:{' '}
              <strong>{workspace.department_name || workspace.department_code || 'N/A'}</strong>
            </p>
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'transparent',
              border: 'none',
              cursor: 'pointer',
              color: 'var(--color-ink-muted, #6b7280)',
              padding: '6px',
              borderRadius: '6px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            ✕
          </button>
        </div>

        {/* Modal Body */}
        <div style={{ padding: '20px 24px', overflowY: 'auto', flex: 1, display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* Add Member Form (Only for OWNER or ADMIN) */}
          {isOwnerOrAdmin && (
            <form
              onSubmit={handleAddMember}
              style={{
                background: 'var(--color-ground, #f9fafb)',
                padding: '16px',
                borderRadius: '12px',
                border: '1px solid var(--color-border, #e5e7eb)',
                display: 'flex',
                gap: '10px',
                alignItems: 'flex-end',
                flexWrap: 'wrap',
              }}
            >
              <div style={{ flex: 1, minWidth: '200px' }}>
                <label
                  style={{
                    display: 'block',
                    fontSize: '11px',
                    fontWeight: 600,
                    textTransform: 'uppercase',
                    color: 'var(--color-ink-muted, #6b7280)',
                    marginBottom: '4px',
                    letterSpacing: '0.05em',
                  }}
                >
                  Add User ID (Dept Member)
                </label>
                <input
                  type="text"
                  placeholder="UUID (e.g. 550e8400-e29b-...)"
                  value={newUserId}
                  onChange={(e) => setNewUserId(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '8px 12px',
                    borderRadius: '8px',
                    border: '1px solid var(--color-border, #d1d5db)',
                    background: '#fff',
                    fontSize: '13px',
                    color: 'var(--color-ink, #111827)',
                  }}
                  required
                />
              </div>

              <div style={{ width: '120px' }}>
                <label
                  style={{
                    display: 'block',
                    fontSize: '11px',
                    fontWeight: 600,
                    textTransform: 'uppercase',
                    color: 'var(--color-ink-muted, #6b7280)',
                    marginBottom: '4px',
                    letterSpacing: '0.05em',
                  }}
                >
                  Role
                </label>
                <select
                  value={newRole}
                  onChange={(e) => setNewRole(e.target.value as WorkspaceRole)}
                  style={{
                    width: '100%',
                    padding: '8px 10px',
                    borderRadius: '8px',
                    border: '1px solid var(--color-border, #d1d5db)',
                    background: '#fff',
                    fontSize: '13px',
                    color: 'var(--color-ink, #111827)',
                  }}
                >
                  <option value="VIEWER">VIEWER</option>
                  <option value="EDITOR">EDITOR</option>
                  <option value="OWNER">OWNER</option>
                </select>
              </div>

              <button
                type="submit"
                disabled={adding || !newUserId.trim()}
                style={{
                  padding: '8px 16px',
                  borderRadius: '8px',
                  border: 'none',
                  background: 'var(--color-teal, #0d9488)',
                  color: '#fff',
                  fontWeight: 600,
                  fontSize: '13px',
                  cursor: adding || !newUserId.trim() ? 'not-allowed' : 'pointer',
                  opacity: adding || !newUserId.trim() ? 0.6 : 1,
                  height: '36px',
                }}
              >
                {adding ? 'Adding…' : '+ Add Member'}
              </button>
            </form>
          )}

          {/* Members List Table */}
          <div>
            <h4
              style={{
                fontSize: '12px',
                fontWeight: 600,
                textTransform: 'uppercase',
                letterSpacing: '0.06em',
                color: 'var(--color-ink-muted, #6b7280)',
                margin: '0 0 10px 0',
              }}
            >
              Current Members ({members.length})
            </h4>

            {loading ? (
              <div style={{ padding: '24px', textAlign: 'center', color: 'var(--color-ink-muted, #6b7280)', fontSize: '13px' }}>
                Loading members…
              </div>
            ) : error ? (
              <div style={{ padding: '16px', background: '#fef2f2', color: '#b91c1c', borderRadius: '8px', fontSize: '13px' }}>
                {error}
              </div>
            ) : members.length === 0 ? (
              <div style={{ padding: '24px', textAlign: 'center', color: 'var(--color-ink-muted, #6b7280)', fontSize: '13px' }}>
                No members found.
              </div>
            ) : (
              <div style={{ border: '1px solid var(--color-border, #e5e7eb)', borderRadius: '10px', overflow: 'hidden' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '13px' }}>
                  <thead>
                    <tr style={{ background: 'var(--color-ground, #f9fafb)', borderBottom: '1px solid var(--color-border, #e5e7eb)' }}>
                      <th style={{ padding: '10px 14px', fontWeight: 600, color: 'var(--color-ink-muted, #4b5563)' }}>User</th>
                      <th style={{ padding: '10px 14px', fontWeight: 600, color: 'var(--color-ink-muted, #4b5563)' }}>Role</th>
                      <th style={{ padding: '10px 14px', fontWeight: 600, color: 'var(--color-ink-muted, #4b5563)', textAlign: 'right' }}>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {members.map((member) => {
                      const badge = ROLE_COLORS[member.role] || ROLE_COLORS.VIEWER;
                      const isSelf = currentUserEmail && member.user_email === currentUserEmail;
                      const isBusy = updatingId === member.user_id;

                      return (
                        <tr
                          key={member.user_id}
                          style={{
                            borderBottom: '1px solid var(--color-border, #e5e7eb)',
                            transition: 'background 0.15s',
                          }}
                        >
                          <td style={{ padding: '12px 14px' }}>
                            <div style={{ fontWeight: 500, color: 'var(--color-ink, #111827)' }}>
                              {member.user_email || member.user_id}
                              {isSelf && (
                                <span
                                  style={{
                                    marginLeft: '6px',
                                    fontSize: '11px',
                                    color: 'var(--color-teal, #0d9488)',
                                    fontWeight: 600,
                                  }}
                                >
                                  (You)
                                </span>
                              )}
                            </div>
                            <div style={{ fontSize: '11px', color: 'var(--color-ink-muted, #9ca3af)', fontFamily: 'monospace' }}>
                              {member.user_id}
                            </div>
                          </td>

                          <td style={{ padding: '12px 14px' }}>
                            {isOwnerOrAdmin ? (
                              <select
                                value={member.role}
                                disabled={isBusy}
                                onChange={(e) => handleRoleChange(member.user_id, e.target.value as WorkspaceRole)}
                                style={{
                                  padding: '4px 8px',
                                  borderRadius: '6px',
                                  fontSize: '12px',
                                  fontWeight: 600,
                                  border: `1px solid ${badge.border}`,
                                  background: badge.bg,
                                  color: badge.color,
                                  cursor: isBusy ? 'wait' : 'pointer',
                                }}
                              >
                                <option value="OWNER">OWNER</option>
                                <option value="EDITOR">EDITOR</option>
                                <option value="VIEWER">VIEWER</option>
                              </select>
                            ) : (
                              <span
                                style={{
                                  display: 'inline-block',
                                  padding: '3px 8px',
                                  borderRadius: '6px',
                                  fontSize: '11px',
                                  fontWeight: 600,
                                  background: badge.bg,
                                  color: badge.color,
                                  border: `1px solid ${badge.border}`,
                                }}
                              >
                                {member.role}
                              </span>
                            )}
                          </td>

                          <td style={{ padding: '12px 14px', textAlign: 'right' }}>
                            {isOwnerOrAdmin && (
                              <button
                                onClick={() => handleRemoveMember(member.user_id)}
                                disabled={isBusy}
                                style={{
                                  padding: '4px 10px',
                                  borderRadius: '6px',
                                  border: '1px solid #fee2e2',
                                  background: '#fff5f5',
                                  color: '#dc2626',
                                  fontSize: '12px',
                                  fontWeight: 500,
                                  cursor: isBusy ? 'wait' : 'pointer',
                                  opacity: isBusy ? 0.5 : 1,
                                }}
                              >
                                Remove
                              </button>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>

        {/* Modal Footer */}
        <div
          style={{
            padding: '14px 24px',
            borderTop: '1px solid var(--color-border, #e5e7eb)',
            display: 'flex',
            justifyContent: 'flex-end',
            background: 'var(--color-ground, #f9fafb)',
          }}
        >
          <button
            onClick={onClose}
            style={{
              padding: '8px 18px',
              borderRadius: '8px',
              border: '1px solid var(--color-border, #d1d5db)',
              background: '#fff',
              fontSize: '13px',
              fontWeight: 500,
              cursor: 'pointer',
              color: 'var(--color-ink, #374151)',
            }}
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
