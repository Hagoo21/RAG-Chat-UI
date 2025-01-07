export const API_URL = 'http://localhost:8000';

export async function uploadFiles(files) {
  const formData = new FormData();
  files.forEach(file => {
    formData.append('files', file);
  });

  try {
    const response = await fetch(`${API_URL}/upload`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Upload failed');
    }

    return response.json();
  } catch (error) {
    console.error('Upload error:', error);
    throw error;
  }
}

export async function sendMessage(message) {
  try {
    const response = await fetch(`${API_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ message })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Chat request failed');
    }

    return response.json();
  } catch (error) {
    console.error('Chat error:', error);
    throw error;
  }
}