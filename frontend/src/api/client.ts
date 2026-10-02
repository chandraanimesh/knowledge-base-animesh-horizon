const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export class ApiError extends Error {
  constructor(public message: string, public status: number, public data?: any) {
    super(message);
    this.name = 'ApiError';
  }
}

export async function apiClient<T>(
  endpoint: string,
  options: RequestInit = {},
  token?: string
): Promise<T> {
  const headers = new Headers(options.headers || {});
  
  if (token) {
    headers.set('Authorization', `Bearer ${token}`);
  }
  
  if (!(options.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  } else if (options.body instanceof FormData) {
    // If it's FormData, let the browser set the Content-Type (which includes the boundary)
    headers.delete('Content-Type');
  }

  const url = `${API_BASE_URL}${endpoint}`;
  const response = await fetch(url, { ...options, headers });

  let data: any;
  try {
    data = await response.json();
  } catch (e) {
    data = await response.text();
  }

  if (!response.ok) {
    const detail = data?.detail || typeof data === 'string' ? data : `HTTP ${response.status}`;
    const errorMessage = typeof detail === 'string' ? detail : JSON.stringify(detail);
    throw new ApiError(errorMessage, response.status, data);
  }

  return data as T;
}
