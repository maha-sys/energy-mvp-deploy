import { getAnalytics, getRecommendations } from './api.js';

(async () => {
  try {
    const data = await getAnalytics();
    const recsResp = await getRecommendations();
    const ul = document.getElementById('insights');
    if (data.error && (!recsResp || recsResp.error)) {
      const li = document.createElement('li');
      li.textContent = 'No insights available.';
      ul.appendChild(li);
      return;
    }

    if (data.trend) {
      const li = document.createElement('li');
      li.textContent = `💡 Trend: ${data.trend}`;
      ul.appendChild(li);
    }

    if (recsResp && recsResp.anomalies && recsResp.anomalies.top_months) {
      Object.entries(recsResp.anomalies.top_months).forEach(([m,v]) => {
        const li = document.createElement('li');
        li.textContent = `💡 Anomaly: ${m} — avg ${Math.round(v)} kWh`;
        ul.appendChild(li);
      });
    } else if (data.peak_usage_impact) {
      Object.entries(data.peak_usage_impact).forEach(([k,v]) => {
        const li = document.createElement('li');
        li.textContent = `💡 Peak ${k}h: avg ${Math.round(v)} kWh`;
        ul.appendChild(li);
      });
    }

    // Top peak hours if present
    if (recsResp && recsResp.anomalies && recsResp.anomalies.top_peak_hours) {
      Object.entries(recsResp.anomalies.top_peak_hours).forEach(([h,v]) => {
        const li = document.createElement('li');
        li.textContent = `💡 Peak ${h}h: avg ${Math.round(v)} kWh`;
        ul.appendChild(li);
      });
    }

  } catch (e) {
    console.error(e);
  }
})();
