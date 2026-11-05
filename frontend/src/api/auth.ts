import { apiClient } from './client';

export async function login(username: string, password: string): Promise<string> {
  const params = new URLSearchParams();
  params.append('username', username);
  params.append('password', password);

  const data = await apiClient<any>('/auth/login', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: params.toString(),
  });
  
  return data.access_token;
}

export async function register(
  email: string,
  password: string,
  requestedDepartment?: string,
  reason?: string
): Promise<any> {
  return await apiClient<any>('/auth/register', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      email,
      password,
      requested_department: requestedDepartment || undefined,
      reason: reason || undefined,
    }),
  });
}
