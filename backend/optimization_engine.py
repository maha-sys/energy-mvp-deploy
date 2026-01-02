import pandas as pd
import numpy as np
from datetime import datetime

class EnergyOptimizer:
    """AI-powered energy optimization engine (dataset aligned)"""

    def optimize(self, df, target_reduction=0.15, time_horizon=30):
        # Baseline
        baseline_units = df['Units_kWh'].mean()
        target_units = baseline_units * (1 - target_reduction)

        # Identify peak usage patterns
        peak_hours_avg = df.groupby('Peak_Usage_Hours')['Units_kWh'].mean()
        high_peak_hours = peak_hours_avg[peak_hours_avg > peak_hours_avg.mean()].index.tolist()

        # Savings estimation
        savings = {
            'peak_shaving': self._peak_shaving(df),
            'load_shifting': self._load_shifting(df),
            'efficiency': self._efficiency(df)
        }

        action_plan = self._action_plan(savings)
        financial_impact = self._financial_impact(df, savings)

        return {
            'current_avg_units_kwh': float(baseline_units),
            'target_avg_units_kwh': float(target_units),
            'target_reduction_percent': target_reduction * 100,
            'high_peak_usage_hours': high_peak_hours,
            'strategies': savings,
            'action_plan': action_plan,
            'financial_impact': financial_impact,
            'timeline_days': time_horizon,
            'generated_at': datetime.now().isoformat()
        }

    def _peak_shaving(self, df):
        peak_avg = df['Peak_Usage_Hours'].mean()
        reduction = df['Units_kWh'].mean() * 0.12

        return {
            'strategy': 'Peak Shaving',
            'estimated_reduction_kwh': float(reduction),
            'savings_percent': 12,
            'difficulty': 'Medium',
            'description': 'Reduce appliance usage during peak hours'
        }

    def _load_shifting(self, df):
        reduction = df['Units_kWh'].mean() * 0.08

        return {
            'strategy': 'Load Shifting',
            'estimated_reduction_kwh': float(reduction),
            'savings_percent': 8,
            'difficulty': 'Easy',
            'description': 'Shift heavy appliance usage to off-peak hours'
        }

    def _efficiency(self, df):
        reduction = df['Units_kWh'].mean() * 0.15

        return {
            'strategy': 'Efficiency Improvements',
            'estimated_reduction_kwh': float(reduction),
            'savings_percent': 15,
            'difficulty': 'Hard',
            'description': 'Use energy-efficient appliances and insulation'
        }

    def _action_plan(self, savings):
        plan = []
        for s in savings.values():
            plan.append({
                'strategy': s['strategy'],
                'priority': 'High' if s['savings_percent'] >= 12 else 'Medium',
                'difficulty': s['difficulty'],
                'expected_savings_percent': s['savings_percent']
            })

        return sorted(plan, key=lambda x: x['expected_savings_percent'], reverse=True)

    def _financial_impact(self, df, savings):
        rate_per_kwh = df['Cost'].sum() / df['Units_kWh'].sum()

        breakdown = {}
        total_monthly = 0

        for s in savings.values():
            monthly = s['estimated_reduction_kwh'] * rate_per_kwh
            breakdown[s['strategy']] = {
                'monthly_savings': float(monthly),
                'annual_savings': float(monthly * 12)
            }
            total_monthly += monthly

        return {
            'currency': 'INR',
            'monthly_savings': float(total_monthly),
            'annual_savings': float(total_monthly * 12),
            'breakdown': breakdown
        }

    """def generate_recommendations(self, df):
        recommendations = []

        avg_units = df['Units_kWh'].mean()
        peak_hours_avg = df['Peak_Usage_Hours'].mean()

        if peak_hours_avg > 6:
            recommendations.append({
                'priority': 'High',
                'title': 'High Peak Hour Usage',
                'description': 'Your peak usage hours are high. Try running appliances during off-peak hours.',
                'potential_savings': '10–15%'
            })

        if avg_units > df['Avg_Daily_KWh'].mean() * 30:
            recommendations.append({
                'priority': 'Medium',
                'title': 'High Monthly Consumption',
                'description': 'Monthly usage exceeds daily average trend.',
                'potential_savings': '8–12%'
            })

        recommendations.append({
            'priority': 'Medium',
            'title': 'Switch to Energy Efficient Devices',
            'description': 'Using 5-star rated appliances reduces long-term consumption.',
            'potential_savings': '15–20%'
        })

        return {
            'recommendations': recommendations,
            'generated_at': datetime.now().isoformat()
        }"""
    def generate_recommendations(self, df):
        recommendations = []

        baseline_monthly = float(df['Units_kWh'].mean())
        avg_daily = float(df['Avg_Daily_KWh'].mean())
        expected_monthly = avg_daily * 30.0

        # Fraction of rows with high peak hours (relative to 75th percentile)
        peak_threshold = df['Peak_Usage_Hours'].quantile(0.75)
        peak_rows = df[df['Peak_Usage_Hours'] >= peak_threshold]
        peak_kwh = float(peak_rows['Units_kWh'].sum())

        # Estimate shiftable energy and expected reduction (domain heuristics)
        shiftable_frac = 0.5
        est_shift_kwh = peak_kwh * shiftable_frac
        est_reduction_kwh = est_shift_kwh * 0.2  # assume 20% of shiftable yields real savings

        rate_per_kwh = (df['Cost'].sum() / df['Units_kWh'].sum()) if df['Units_kWh'].sum() > 0 else None
        est_cost_savings = (est_reduction_kwh * rate_per_kwh) if rate_per_kwh is not None else None

        # Peak-shaving recommendation
        if est_reduction_kwh > baseline_monthly * 0.03:  # meaningful threshold (>3% baseline)
            recommendations.append({
                'priority': 'High',
                'title': 'Peak-shaving (shift loads)',
                'description': f'{len(peak_rows)} records indicate concentrated peak usage (≥ {peak_threshold}).',
                'estimated_kwh_savings': round(est_reduction_kwh, 2),
                'estimated_monthly_savings_inr': round(est_cost_savings, 2) if est_cost_savings is not None else None,
                'potential_savings': f"{int((est_reduction_kwh/baseline_monthly)*100)}%",
                'confidence': 'medium'
            })

        # Efficiency recommendations based on baseline and cost
        est_eff_pct = 0.12 if baseline_monthly > 250 else 0.08
        est_eff_kwh = baseline_monthly * est_eff_pct
        est_eff_cost = est_eff_kwh * rate_per_kwh if rate_per_kwh is not None else None
        if est_eff_kwh > 0:
            recommendations.append({
                'priority': 'Medium',
                'title': 'Efficiency improvements',
                'description': 'Energy-efficient appliances and insulation to reduce consumption.',
                'estimated_kwh_savings': round(est_eff_kwh, 2),
                'estimated_monthly_savings_inr': round(est_eff_cost, 2) if est_eff_cost is not None else None,
                'potential_savings': f"{int(est_eff_pct*100)}%",
                'confidence': 'low'
            })

        # High monthly anomaly detection
        anomalies = {}
        try:
            monthly_avg = df.groupby('Month')['Units_kWh'].mean()
            top_months = monthly_avg.sort_values(ascending=False).head(3).to_dict()
            anomalies['top_months'] = {str(k): float(v) for k, v in top_months.items()}
        except Exception:
            anomalies['top_months'] = {}

        # Peak hour ranking
        try:
            peak_by_hour = df.groupby('Peak_Usage_Hours')['Units_kWh'].mean().sort_values(ascending=False).head(5).to_dict()
            anomalies['top_peak_hours'] = {str(int(k)): float(v) for k, v in peak_by_hour.items()}
        except Exception:
            anomalies['top_peak_hours'] = {}

        return {
            'recommendations': recommendations,
            'anomalies': anomalies,
            'generated_at': datetime.now().isoformat()
        }
    
