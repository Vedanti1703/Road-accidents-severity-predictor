// src/components/Dashboard.js
import React from "react";

export default function Dashboard() {
  return (
    <div className="dashboard-page">
      <h1>Data Insights Dashboard</h1>

      <section>
        <h2>Accident Severity Distribution</h2>
        <img src="http://localhost:5000/static_charts/severity_distribution.png" alt="Severity Distribution" />
      </section>

     
      <section>
        <h2>Severity by Time Period</h2>
        <img src="http://localhost:5000/static_charts/time_period_severity.png" alt="Severity by Time" />
      </section>
      


      <section>
        <h2>Number of Casualties by Severity</h2>
        <img src="http://localhost:5000/static_charts/casualties_severity.png" alt="Casualties Severity" />
      </section>
    </div>
  );
}
