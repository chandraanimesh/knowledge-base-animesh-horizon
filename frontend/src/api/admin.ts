import { apiClient } from './client';

export interface UserResponse {
  id: string;
  email: string;
  role: string;
  is_active: boolean;
  created_at: string;
}

export interface DepartmentResponse {
  id: string;
  code: string;
  name: string;
  is_active: boolean;
  created_at?: string;
}

export interface PermissionResponse {
  id: string;
  code: string;
  description?: string;
  is_active: boolean;
}

export interface DeptAccessSummary {
  code: string;
  name: string;
  permissions: string[];
}

export interface UserAccessSummary {
  id: string;
  email: string;
  role: string;
  is_active: boolean;
  created_at: string;
  departments: DeptAccessSummary[];
}

export interface CreateUserPayload {
  email: string;
  password: string;
  role: string;
  department_code?: string;
  permissions?: string[];
}

export const adminApi = {
  getUsers: (token: string) => 
    apiClient<UserResponse[]>('/admin/users', { method: 'GET' }, token),
  
  getUsersWithAccess: (token: string) =>
    apiClient<UserAccessSummary[]>('/admin/users/with-access', { method: 'GET' }, token),

  createUser: (data: CreateUserPayload, token: string) => 
    apiClient<UserResponse>('/admin/users', { 
      method: 'POST', 
      body: JSON.stringify(data) 
    }, token),

  getUser: (userId: string, token: string) => 
    apiClient<UserResponse>(`/admin/users/${userId}`, { method: 'GET' }, token),

  updateUserStatus: (userId: string, isActive: boolean, token: string) => 
    apiClient<UserResponse>(`/admin/users/${userId}/status`, { 
      method: 'PATCH', 
      body: JSON.stringify({ is_active: isActive }) 
    }, token),

  getUserDepartments: (userId: string, token: string) => 
    apiClient<DepartmentResponse[]>(`/admin/users/${userId}/departments`, { method: 'GET' }, token),

  assignUserDepartment: (userId: string, deptCode: string, token: string) => 
    apiClient<any>(`/admin/users/${userId}/departments/${deptCode}`, { method: 'POST' }, token),

  revokeUserDepartment: (userId: string, deptCode: string, token: string) => 
    apiClient<any>(`/admin/users/${userId}/departments/${deptCode}`, { method: 'DELETE' }, token),

  getUserPermissions: (userId: string, deptCode: string, token: string) => 
    apiClient<PermissionResponse[]>(`/admin/users/${userId}/departments/${deptCode}/permissions`, { method: 'GET' }, token),

  grantUserPermission: (userId: string, deptCode: string, permCode: string, token: string) => 
    apiClient<any>(`/admin/users/${userId}/departments/${deptCode}/permissions/${permCode}`, { method: 'POST' }, token),

  revokeUserPermission: (userId: string, deptCode: string, permCode: string, token: string) => 
    apiClient<any>(`/admin/users/${userId}/departments/${deptCode}/permissions/${permCode}`, { method: 'DELETE' }, token),

  applyPermissionPreset: (userId: string, deptCode: string, preset: string, token: string) => 
    apiClient<any>(`/admin/users/${userId}/departments/${deptCode}/permission-preset`, { 
      method: 'POST', 
      body: JSON.stringify({ preset }) 
    }, token),

  listDepartments: (token: string) =>
    apiClient<DepartmentResponse[]>('/admin/users/departments', { method: 'GET' }, token),

  getAllDepartments: (token: string) =>
    apiClient<DepartmentResponse[]>('/admin/departments', { method: 'GET' }, token),

  createDepartment: (data: { code: string; name: string }, token: string) =>
    apiClient<DepartmentResponse>('/admin/departments', {
      method: 'POST',
      body: JSON.stringify(data),
    }, token),

  updateDepartmentStatus: (code: string, isActive: boolean, token: string) =>
    apiClient<DepartmentResponse>(`/admin/departments/${code}/status`, {
      method: 'PATCH',
      body: JSON.stringify({ is_active: isActive }),
    }, token),
};
