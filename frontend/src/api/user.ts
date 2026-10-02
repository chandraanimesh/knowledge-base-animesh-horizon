import { apiClient } from './client';
import { UserResponse } from './admin';

export const userApi = {
  getMe: (token: string) => 
    apiClient<UserResponse>('/auth/me', { method: 'GET' }, token),
  
  getMyAccess: (token: string) => 
    apiClient<Record<string, string[]>>('/auth/me/access', { method: 'GET' }, token),
};
