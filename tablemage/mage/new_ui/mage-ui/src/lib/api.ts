const API_BASE_URL = 'http://127.0.0.1:5005/api';

export interface AnalysisItem {
  file_type: 'figure' | 'table' | 'thought' | 'code';
  file_name?: string;
  file_path?: string;
  content?: string;
}

export interface UploadResponse {
  message: string;
  rows: number;
  columns: string[];
  preview: Record<string, any>;
}
export const sendChatMessage = async (message: string): Promise<{
    response: string;
    hasAnalysis: boolean;
  }> => {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message }),
    });
  
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to send message');
    }
  
    return response.json();
  };
export const uploadDataset = async (file: File): Promise<UploadResponse> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to upload dataset');
  }

  return response.json();
};

export const getAnalysisHistory = async (): Promise<AnalysisItem[]> => {
  const response = await fetch(`${API_BASE_URL}/analysis`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to fetch analysis history');
  }

  return response.json();
};

export const healthCheck = async (): Promise<{ status: string; message: string }> => {
  const response = await fetch(`${API_BASE_URL}/health`);

  if (!response.ok) {
    throw new Error('Failed to connect to server');
  }

  return response.json();
};