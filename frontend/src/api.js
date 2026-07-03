const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  let body = null;
  try {
    body = await res.json();
  } catch {
    // no body / not JSON
  }

  if (!res.ok) {
    const detail =
      (body && (body.detail || JSON.stringify(body))) || `HTTP ${res.status}`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }

  return body;
}

export const api = {
  startNegotiation: (payload) =>
    request("/negotiation/start", { method: "POST", body: JSON.stringify(payload) }),

  step: (sessionId, payload) =>
    request(`/negotiation/${sessionId}/step`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  end: (sessionId, payload) =>
    request(`/negotiation/${sessionId}/end`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  getState: (sessionId) => request(`/negotiation/${sessionId}/state`),

  getClientHistory: (clientId) => request(`/clients/${clientId}/history`),
};
