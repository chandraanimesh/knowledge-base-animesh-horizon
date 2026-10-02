import { apiClient } from './client';

export type PermissionCode = 'VIEW' | 'UPLOAD' | 'EDIT' | 'DELETE';
export type AccessRequestStatus = 'PENDING' | 'APPROVED' | 'REJECTED' | 'CANCELLED';

export interface AccessRequestCreatePayload {
  department_code: string;
  requested_permissions: PermissionCode[];
  reason?: string;
}

export interface AccessRequestApprovePayload {
  approved_permissions?: PermissionCode[];
  review_note?: string;
}

export interface AccessRequestRejectPayload {
  review_note?: string;
}

export interface AccessRequestResponse {
  id: string;
  user_id: string;
  user_email: string;
  department_id: string;
  department_code: string;
  department_name: string;
  requested_permissions: PermissionCode[];
  status: AccessRequestStatus;
  reason?: string | null;
  requested_at: string;
  reviewed_at?: string | null;
  reviewed_by?: string | null;
  reviewer_email?: string | null;
  review_note?: string | null;
}

export async function createAccessRequest(
  payload: AccessRequestCreatePayload,
  token: string
): Promise<AccessRequestResponse> {
  return apiClient<AccessRequestResponse>('/access-requests', {
    method: 'POST',
    body: JSON.stringify(payload),
  }, token);
}

export async function getMyAccessRequests(
  token: string
): Promise<AccessRequestResponse[]> {
  return apiClient<AccessRequestResponse[]>('/access-requests/my', {
    method: 'GET',
  }, token);
}

export async function getAccessRequests(
  token: string,
  statusFilter?: AccessRequestStatus
): Promise<AccessRequestResponse[]> {
  const query = statusFilter ? `?status_filter=${encodeURIComponent(statusFilter)}` : '';
  return apiClient<AccessRequestResponse[]>(`/access-requests${query}`, {
    method: 'GET',
  }, token);
}

export async function approveAccessRequest(
  id: string,
  permissionsOrPayload: PermissionCode[] | AccessRequestApprovePayload | undefined,
  reviewNoteOrToken: string | undefined,
  token?: string
): Promise<AccessRequestResponse> {
  let payload: AccessRequestApprovePayload;
  let authToken: string;

  if (permissionsOrPayload && !Array.isArray(permissionsOrPayload) && typeof permissionsOrPayload === 'object') {
    payload = permissionsOrPayload;
    authToken = reviewNoteOrToken || '';
  } else {
    payload = {
      approved_permissions: permissionsOrPayload,
      review_note: reviewNoteOrToken,
    };
    authToken = token || '';
  }

  return apiClient<AccessRequestResponse>(`/access-requests/${encodeURIComponent(id)}/approve`, {
    method: 'POST',
    body: JSON.stringify(payload),
  }, authToken);
}

export async function rejectAccessRequest(
  id: string,
  reviewNoteOrPayload: string | AccessRequestRejectPayload | undefined,
  token?: string
): Promise<AccessRequestResponse> {
  let payload: AccessRequestRejectPayload;
  let authToken: string;

  if (typeof reviewNoteOrPayload === 'object' && reviewNoteOrPayload !== null) {
    payload = reviewNoteOrPayload;
    authToken = token || '';
  } else {
    payload = { review_note: reviewNoteOrPayload };
    authToken = token || '';
  }

  return apiClient<AccessRequestResponse>(`/access-requests/${encodeURIComponent(id)}/reject`, {
    method: 'POST',
    body: JSON.stringify(payload),
  }, authToken);
}

export const accessRequestApi = {
  createAccessRequest,
  getMyAccessRequests,
  getAccessRequests,
  approveAccessRequest,
  rejectAccessRequest,
};
