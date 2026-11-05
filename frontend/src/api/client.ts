const rawApiUrl = import.meta.env.VITE_API_BASE_URL || '';
// Avoid mixed-content blocking: if page is on HTTPS, never fetch insecure http://
const isHttps = typeof window !== 'undefined' && window.location.protocol === 'https:';
const API_BASE_URL = (isHttps && rawApiUrl.startsWith('http://')) ? '' : rawApiUrl;

export class ApiError extends Error {
  status: number;
  data?: any;

  constructor(message: string, status: number, data?: any) {
    super(message);
    this.status = status;
    this.data = data;
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

  const rawText = await response.text();
  let data: any;
  try {
    data = rawText ? JSON.parse(rawText) : null;
  } catch {
    data = rawText;
  }

  if (!response.ok) {
    const detail = data?.detail || (typeof data === 'string' && data.trim().length > 0 ? data : `HTTP ${response.status}`);
    const errorMessage = typeof detail === 'string' ? detail : JSON.stringify(detail);
    throw new ApiError(errorMessage, response.status, data);
  }

  return data as T;
}
