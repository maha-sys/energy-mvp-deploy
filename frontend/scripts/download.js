import { getAnalytics, getRecommendations, API_BASE_URL } from './api.js';

export async function downloadCSV(){
  console.log('[downloadCSV] triggered');
  try {
    const analytics = await getAnalytics();
    const recsRes = await getRecommendations();
    console.log('[downloadCSV] analytics', analytics);
    console.log('[downloadCSV] recommendations', recsRes);

    const lines = [];
    lines.push(['Metric','Value']);
    lines.push(['Total Units', analytics.total_units ?? 'N/A']);
    lines.push(['Total Cost', analytics.total_cost ?? 'N/A']);
    lines.push(['Trend', analytics.trend ?? 'N/A']);
    lines.push([]);

    lines.push(['Monthly','Avg_Units_kWh']);
    const monthly = analytics.monthly_avg_units || {};
    for (const [m, v] of Object.entries(monthly)) {
      lines.push([m, v === null ? 'N/A' : v]);
    }
    lines.push([]);

    lines.push(['Recommendations','Priority','Potential Savings']);
    const recs = (recsRes && (recsRes.recommendations || recsRes)) || [];
    if (Array.isArray(recs) && recs.length) {
      recs.forEach(r => {
        lines.push([r.title || r.strategy || '', r.priority || '', r.potential_savings || '']);
      });
    } else {
      lines.push(['No recommendations available','','']);
    }

    // Convert to CSV string
    const csv = lines.map(r => r.map(cell => {
      if (cell === null || cell === undefined) return '';
      const s = String(cell);
      // escape quotes
      return s.includes(',') || s.includes('"') || s.includes('\n') ? '"' + s.replace(/"/g, '""') + '"' : s;
    }).join(',')).join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'energy_report.csv';
    a.click();
  } catch (e) {
    alert('Failed to generate report: ' + (e.message || e));
    console.error(e);
  }
}

export async function downloadPDF(){
  console.log('[downloadPDF] triggered');
  try {
    const res = await fetch(`${API_BASE_URL}/reports/pdf`, { headers: { 'Accept': 'application/pdf' } });
    console.log('[downloadPDF] fetch response', res);

    if (res.ok) {
      const contentType = (res.headers.get('content-type') || '');
      console.log('[downloadPDF] content-type:', contentType);
      if (contentType.includes('application/pdf')) {
        const blob = await res.blob();
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'energy_report.pdf';
        a.click();
        console.log('[downloadPDF] downloaded via server');
        return;
      }

      // not a PDF - maybe JSON telling us to fallback
      const json = await res.json().catch(() => null);
      console.warn('[downloadPDF] not pdf, server json', json);
      if (json && json.error === 'pdf_unavailable') {
        // server cannot create PDF -> client-side fallback
        await generateClientPdf();
        return;
      }

      throw new Error(json?.message || 'Unexpected response from server');
    } else {
      const json = await res.json().catch(() => null);
      console.warn('[downloadPDF] non-ok response', res.status, json);
      if (json && json.error === 'pdf_unavailable') {
        await generateClientPdf();
        return;
      }
      throw new Error(json?.message || `Failed to fetch PDF (${res.status})`);
    }
  } catch (e) {
    console.error('[downloadPDF] PDF generation failed:', e);
    // final fallback: try client-side generation
    try {
      await generateClientPdf();
    } catch (err) {
      alert('Failed to generate PDF: ' + (err.message || err));
    }
  }
}

async function generateClientPdf(){
  // dynamically load jsPDF if not present
  if (!window.jspdf) {
    await new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = 'https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js';
      s.onload = resolve;
      s.onerror = () => reject(new Error('Failed to load jsPDF'));
      document.head.appendChild(s);
    });
  }

  const { jsPDF } = window.jspdf;
  const doc = new jsPDF();
  doc.setFontSize(16);
  doc.text('EnergyAI Optimization Report', 10, 20);
  doc.setFontSize(10);
  doc.text('Generated: ' + new Date().toISOString(), 10, 28);

  const analytics = await getAnalytics();
  const recsRes = await getRecommendations();

  let y = 40;
  doc.text(`Total Units: ${analytics.total_units ?? 'N/A'}`, 10, y); y += 6;
  doc.text(`Total Cost: ${analytics.total_cost ?? 'N/A'}`, 10, y); y += 10;

  doc.text('Monthly Avg Units:', 10, y); y += 6;
  const monthly = analytics.monthly_avg_units || {};
  for (const [m, v] of Object.entries(monthly)) {
    doc.text(`${m}: ${v}`, 12, y); y += 6;
    if (y > 270) { doc.addPage(); y = 20; }
  }

  y += 6;
  doc.text('Recommendations:', 10, y); y += 6;
  const recs = (recsRes && (recsRes.recommendations || recsRes)) || [];
  if (!recs.length) {
    doc.text('No recommendations available', 12, y);
  } else {
    recs.forEach(r => {
      doc.text(`- ${r.title || r.strategy || ''} ${r.estimated_monthly_savings_inr ? '(₹'+r.estimated_monthly_savings_inr+')' : ''}`, 12, y);
      y += 6;
      if (r.description) { doc.text(r.description.slice(0, 200), 14, y); y += 6; }
      if (y > 270) { doc.addPage(); y = 20; }
    });
  }

  doc.save('energy_report.pdf');
}

// Keep existing default for backwards compat
export default downloadCSV;
