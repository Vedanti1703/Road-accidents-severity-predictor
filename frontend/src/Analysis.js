// src/Analysis.js
import React, { useState, useEffect, useMemo, useRef } from "react";
import { useLocation, Link } from "react-router-dom";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
  Legend
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend);

// -------------------- THEME --------------------
const THEME = {
  bg: "#0f1112",
  card: "#121416",
  text: "#e6eef2",
  subtext: "#9fb3bd",
  accent: "#1fb5a1",   
  danger: "#e14b4b"    
};

ChartJS.defaults.color = THEME.text;
ChartJS.defaults.font.size = 13;
ChartJS.defaults.plugins.legend.display = false;

// -------------------- OPTIONS --------------------
const AGE_OPTIONS = ["Under 18", "18-30", "31-50", "Over 51", "Unknown"];
const EXP_OPTIONS = ["Below 1yr", "1-2yr", "2-5yr", "5-10yr", "Above 10yr", "No Licence", "unknown"];
const VEHICLE_OPTIONS = [
  "Automobile","Bajaj","Bicycle","Long lorry","Lorry (11?40Q)","Lorry (41?100Q)",
  "Motorcycle","Other","Pick up upto 10Q","Public (12 seats)","Public (13?45 seats)",
  "Public (> 45 seats)","Ridden horse","Special vehicle","Stationwagen","Taxi","Turbo"
];

// -------------------- DEBOUNCE --------------------
function useDebouncedCallback(fn, delay = 350) {
  const ref = useRef();
  return (...args) => {
    clearTimeout(ref.current);
    ref.current = setTimeout(() => fn(...args), delay);
  };
}

export default function Analysis() {

  const { state } = useLocation();
  const initialResult = state?.result || null;

  const [result, setResult] = useState(initialResult);
  const [loadingSim, setLoadingSim] = useState(false);
  const [error, setError] = useState(null);

  const inputSummary = result?.input_summary || {};

  const [whatIf, setWhatIf] = useState({
    Age_band_of_driver: inputSummary.driver_age || AGE_OPTIONS[1],
    Driving_experience: inputSummary.experience || EXP_OPTIONS[1],
    Type_of_vehicle: inputSummary.vehicle_type || VEHICLE_OPTIONS[0]
  });

  // -------------------- SIMULATE --------------------
  const simulate = useDebouncedCallback(async (payload) => {
    try {
      setLoadingSim(true);
      const res = await fetch("http://localhost:5000/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
      setResult(data);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoadingSim(false);
    }
  }, 350);

  useEffect(() => { simulate(whatIf); }, [whatIf]);

  // -------------------- DATA --------------------
  const topFeatures = result?.explanation?.top_features || [];
  const probabilities = result?.probabilities || {};

  const originalTop = useMemo(() => {
    return (initialResult?.explanation?.top_features || []).reduce((a, f) => {
      a[f.feature] = f.impact;
      return a;
    }, {});
  }, [initialResult]);

  const trendData = useMemo(() => {
    return topFeatures.map(f => {
      const prev = originalTop[f.feature] ?? 0;
      const diff = f.impact - prev;

      return {
        feature: f.feature,
        impact: f.impact,
        arrow: diff > 0.03 ? "↑" : diff < -0.03 ? "↓" : "→",
        color: diff > 0.03 ? THEME.danger : diff < -0.03 ? THEME.accent : THEME.subtext
      };
    });
  }, [topFeatures, originalTop]);

  // -------------------- CHARTS --------------------
  const probBarData = useMemo(() => ({
    labels: Object.keys(probabilities),
    datasets: [{
      label: "Probability (%)",
      data: Object.values(probabilities),
      backgroundColor: THEME.accent,
      borderRadius: 6
    }]
  }), [probabilities]);

  const featBarData = useMemo(() => ({
    labels: topFeatures.map(f => f.feature),
    datasets: [{
      label: "Feature Contribution",
      data: topFeatures.map(f => f.impact),
      backgroundColor: topFeatures.map(f => f.impact >= 0 ? THEME.accent : THEME.danger),
      borderRadius: 6
    }]
  }), [topFeatures]);

  const card = { background: THEME.card, padding: 18, borderRadius: 10 };
// ---- Generate Conclusion Text from Features ----
function generateConclusion(severity, topFeatures) {
  if (!severity || !topFeatures || topFeatures.length === 0) {
    return "Insights are being generated...";
  }

  const top = topFeatures.slice(0, 3).map(f => f.feature).join(", ");

  if (severity === "Fatal injury") {
    return `The model predicts a highly critical accident severity. The most influential factors contributing to this high severity are: ${top}. Immediate safety interventions are advised.`;
  }

  if (severity === "Serious Injury") {
    return `The accident is likely to result in serious injury. Key contributing factors include: ${top}. Encouraging safer driving strategies and improved road safety awareness could reduce risk.`;
  }

  return `The model predicts moderate accident severity. The primary influencing factors are: ${top}. Risk can be reduced through cautious driving and adherence to road safety measures.`;
}


  // -------------------- UI --------------------
  return (
    <div style={{ background: THEME.bg, minHeight: "100vh", padding: 28, color: THEME.text }}>

      {!initialResult ? (
        <div style={{ textAlign: "center", marginTop: 80 }}>
          <h2>No Analysis Data</h2>
          <Link to="/"><button>← Back</button></Link>
        </div>
      ) : (
      <div style={{ maxWidth: 1200, margin: "auto" }}>

        <h1 style={{ color: THEME.accent }}>Accident Severity Analysis</h1>

        {/* Charts */}
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:20, marginTop:20 }}>
          <div style={card}><div style={{height:260}}><Bar data={probBarData} options={{maintainAspectRatio:false}}/></div></div>
          <div style={card}><div style={{height:260}}><Bar data={featBarData} options={{indexAxis:"y", maintainAspectRatio:false}}/></div></div>
        </div>
                {/* Heatmap */}
        {/* HEATMAP (Image-based) */}
<div style={{ ...card, marginTop: 20 }}>
  <h3>Top Feature Interaction Heatmap</h3><br></br>
  <img 
    src="http://localhost:5000/static/heatmap.png" 
    alt="Feature Heatmap"
    style={{ width: "70%", borderRadius: "10px" }}
  />
</div>


        {/* Histogram: Accidents by Hour */}
{/* TIME PERIOD HISTOGRAM */}
<div style={{ ...card, marginTop: 20 }}>
  <h3>Accident Frequency by Time of Day(Histogram)</h3>
  <div style={{ height: 260 }}>
    <Bar
      data={{
        labels: result.time_distribution?.labels || [],
        datasets: [{
          label: "Accident Count",
          data: result.time_distribution?.values || [],
          backgroundColor:"#ffb6c1",
          borderRadius: 6
        }]
      }}
      options={{ maintainAspectRatio: false }}
    />
  </div>
</div>


        {/* Trend + Summary */}
        <div style={{ display:"grid", gridTemplateColumns:"2fr 1fr", gap:20, marginTop:20 }}>
          <div style={card}>
            <h3>Live Risk Trend (What-If Impact)</h3>
            <div style={{height:260}}>
              <Bar data={{
                labels: trendData.map(f => `${f.arrow} ${f.feature}`),
                datasets: [{ data: trendData.map(f => f.impact), backgroundColor: trendData.map(f => f.color), borderRadius:6 }]
              }} options={{ indexAxis:"y", maintainAspectRatio:false }} />
            </div>
          </div>

          <div style={card}>
            <h3>Scenario Summary</h3>
            <p style={{ color: THEME.subtext, lineHeight: 1.6 }}>
              <strong>Driver Age:</strong> {inputSummary.driver_age}<br/>
              <strong>Experience:</strong> {inputSummary.experience}<br/>
              <strong>Vehicle:</strong> {inputSummary.vehicle_type}
            </p>

            {/* ✅ CONCLUSION ADDED */}
            <div style={{ marginTop:12, padding:"10px 14px", background:"#1a1d1f", borderRadius:8 }}>
              <strong>Conclusion:</strong><br/>
              {generateConclusion(result?.prediction, topFeatures)}
            </div>
          </div>
        </div>
         {/* SAFETY RECOMMENDATIONS */}
{/* ✅ Recommendations Section */}
{/* Safety Recommendations */}
<div style={{ ...card, marginTop: 20 }}>
  <h3>Safety Recommendations</h3>

{result?.recommendations?.length > 0 && (
  <div style={{ ...card, marginTop: 25 }}>
    <h3>Safety Recommendations</h3>
    <table style={{ width: "100%", borderCollapse: "collapse", marginTop: 10 }}>
      <thead>
        <tr>
          <th>Factor</th>
          <th>Value</th>
          <th>Risk</th>
          <th>Recommendation</th>
        </tr>
      </thead>
    <tbody>
        {result.recommendations.map((rec, i) => (
          <tr key={i}>
            <td>{rec.factor}</td>
            <td style={{ color: THEME.text }}>{rec.value}</td>
            <td style={{ color: rec?.risk?.includes("High") ? THEME.danger : THEME.accent }}>
              {rec.risk}
            </td>
            <td style={{ color: THEME.subtext }}>{rec.recommendation}</td>
          </tr>
        ))}
      </tbody>
    </table>
  </div>
)}
</div>

        <Link to="/"><button style={{ marginTop:20 }} className="btn btn--primary btn--lg">← Back</button></Link>
      </div>
      )}
    </div>
  );
}
