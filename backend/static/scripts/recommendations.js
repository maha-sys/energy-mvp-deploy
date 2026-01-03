import { getRecommendations } from './api.js';

(async () => {
  try {
    const res = await getRecommendations();
    const container = document.getElementById('recs');
    if (!res || res.error) {
      container.textContent = 'No recommendations available.';
      return;
    }

    // Normalize to an array
    let recs = Array.isArray(res.recommendations) ? res.recommendations : (Array.isArray(res) ? res : []);

    if (!recs.length) {
      container.textContent = 'No recommendations available.';
      return;
    }

    // Sort by estimated_monthly_savings_inr desc if available
    recs.sort((a,b) => (b.estimated_monthly_savings_inr || 0) - (a.estimated_monthly_savings_inr || 0));

    recs.forEach(r => {
      const card = document.createElement('div');
      card.style.marginBottom = '12px';
      card.style.paddingBottom = '8px';
      card.innerHTML = `
        <h4 style="margin:0">${r.title || 'Recommendation'}</h4>
        <p style="margin:6px 0;color:var(--text-muted)">${r.description || ''}</p>
        <div style="display:flex;gap:12px;align-items:center;color:var(--text-muted)">
          <small>Priority: <strong style="color:inherit">${r.priority || 'Medium'}</strong></small>
          <small>Potential: <strong>${r.potential_savings || ''}</strong></small>
          ${r.estimated_kwh_savings ? `<small>Est kWh/month: <strong>${r.estimated_kwh_savings}</strong></small>` : ''}
          ${r.estimated_monthly_savings_inr ? `<small>Est INR/month: <strong>₹${r.estimated_monthly_savings_inr}</strong></small>` : ''}
          ${r.confidence ? `<small style="background:rgba(255,255,255,0.05);padding:2px 6px;border-radius:6px">${r.confidence}</small>` : ''}
        </div>
      `;
      container.appendChild(card);
    });

  } catch (e) {
    console.error(e);
    const container = document.getElementById('recs');
    container.textContent = 'Failed to load recommendations.';
  }
})();"<p>✔ Reduce AC usage by 1 hour/day</p>";
