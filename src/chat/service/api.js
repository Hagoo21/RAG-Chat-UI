export const API_URL = process.env.NODE_ENV === 'production' 
  ? 'https://chat-mim-backend.onrender.com'  // Update this with your backend URL after deployment
  : 'http://localhost:8000';

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

export async function getIncidents() {
  try {
    const response = await fetch(`${API_URL}/incidents`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      }
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Fetch incidents error:', error);
    throw new Error('Failed to fetch incidents.');
  }
}

export async function getContext(message) {
  try {
    const response = await fetch(`${API_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify({ message })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log('Context API response:', data);
    return data;
  } catch (error) {
    console.error('Chat context error:', error);
    throw new Error('Failed to fetch chat context.');
  }
}