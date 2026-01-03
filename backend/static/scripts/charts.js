import { getAnalytics } from './api.js';

const ctx = document.getElementById('energyChart').getContext('2d');
const chart = new Chart(ctx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [{
      label: 'Energy Usage (kWh)',
      data: [],
      borderColor: '#38bdf8',
      backgroundColor: 'rgba(56,189,248,0.2)',
      tension: 0.45,
      fill: true
    }]
  },
  options: {
    plugins: { legend: { labels: { color: '#e5e7eb' } } },
    scales: { x: { ticks: { color: '#9ca3af' } }, y: { ticks: { color: '#9ca3af' } } }
  }
});

async function loadAnalytics() {
  try {
    const data = await getAnalytics();
    if (data.error) {
      console.error('Analytics error', data.error);
      return;
    }

    const monthly = data.monthly_avg_units || {};
    chart.data.labels = Object.keys(monthly);
    chart.data.datasets[0].data = Object.values(monthly);
    chart.update();

    // Update cards
    const cards = document.querySelectorAll('.grid .glass');
    if (cards && Object.values(monthly).length) {
      const latest = Object.values(monthly).slice(-1)[0];
      cards[0].innerHTML = `Predicted Usage<br><b>${Math.round(latest || 0)} kWh</b>`;
    }

    if (data.total_units) {
      cards[1].innerHTML = `Estimated Savings<br><b>${Math.round((data.trend === 'decreasing') ? 20 : 10)}%</b>`;
    }

    cards[2].innerHTML = `Efficiency Score<br><b>${data.total_units && data.total_units > 1000 ? 'High' : 'Medium'}</b>`;

  } catch (e) {
    console.error(e);
  }
}

loadAnalytics();
