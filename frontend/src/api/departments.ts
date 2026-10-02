import { apiClient } from './client';
import { DepartmentResponse } from './admin';

export async function getAllDepartments(token: string): Promise<DepartmentResponse[]> {
  return apiClient<DepartmentResponse[]>('/admin/departments', { method: 'GET' }, token);
}

export async function createDepartment(
  code: string,
  name: string,
  token: string
): Promise<DepartmentResponse> {
  return apiClient<DepartmentResponse>('/admin/departments', {
    method: 'POST',
    body: JSON.stringify({ code, name }),
  }, token);
}

export async function updateDepartmentStatus(
  code: string,
  isActive: boolean,
  token: string
): Promise<DepartmentResponse> {
  return apiClient<DepartmentResponse>(`/admin/departments/${encodeURIComponent(code)}/status`, {
    method: 'PATCH',
    body: JSON.stringify({ is_active: isActive }),
  }, token);
}

export async function getActiveDepartments(token: string): Promise<DepartmentResponse[]> {
  return apiClient<DepartmentResponse[]>('/departments', { method: 'GET' }, token);
}

export const departmentApi = {
  getAllDepartments,
  getActiveDepartments,
  createDepartment: (
    codeOrData: string | { code: string; name: string },
    nameOrToken?: string,
    optionalToken?: string
  ) => {
    if (typeof codeOrData === 'object') {
      return createDepartment(codeOrData.code, codeOrData.name, nameOrToken as string);
    }
    return createDepartment(codeOrData, nameOrToken as string, optionalToken as string);
  },
  updateDepartmentStatus,
};
