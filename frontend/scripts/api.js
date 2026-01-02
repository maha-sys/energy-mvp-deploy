export const API_BASE_URL = window.API_BASE_URL || 'http://localhost:5000';

export async function uploadFile(file) {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${API_BASE_URL}/upload`, { method: 'POST', body: form });
  return res.json();
}

export async function predictUsage(payload) {
  const res = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  return res.json();
}

export async function getAnalytics() {
  const res = await fetch(`${API_BASE_URL}/analytics`);
  return res.json();
}

export async function getRecommendations() {
  const res = await fetch(`${API_BASE_URL}/recommendations`);
  return res.json();
}

export async function optimize(payload) {
  const res = await fetch(`${API_BASE_URL}/optimize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  return res.json();
}
