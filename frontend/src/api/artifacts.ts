import { apiClient } from './client.ts';

export interface ArtifactPreviewResponse {
  artifact_id: string;
  title: string;
  format: string;
  mime_type: string;
  content: string;
  byte_size: number;
  checksum_sha256: string;
  is_truncated: boolean;
  is_binary: boolean;
  expires_at?: string | null;
}

export async function getArtifactPreview(
  artifactId: string,
  token: string,
  maxChars: number = 50000
): Promise<ArtifactPreviewResponse> {
  return apiClient<ArtifactPreviewResponse>(
    `/reports/${artifactId}/preview?max_chars=${maxChars}`,
    { method: 'GET' },
    token
  );
}

