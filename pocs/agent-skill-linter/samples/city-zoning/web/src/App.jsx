import React, { useEffect, useState } from 'react';
import { fetchZones, evaluate } from './api.js';

const DEFAULT_PROPOSAL = {
  use: 'single_family',
  heightFeet: 30,
  footprintArea: 1500,
  lotArea: 6000,
  floorArea: 2500,
  frontSetback: 30,
  sideSetback: 10,
  rearSetback: 30
};

const NUMERIC_FIELDS = [
  'heightFeet', 'footprintArea', 'lotArea', 'floorArea',
  'frontSetback', 'sideSetback', 'rearSetback'
];

export default function App() {
  const [zones, setZones] = useState({});
  const [zone, setZone] = useState('R1');
  const [proposal, setProposal] = useState(DEFAULT_PROPOSAL);
  const [result, setResult] = useState(null);

  useEffect(() => {
    fetchZones().then(setZones).catch(() => setZones({}));
  }, []);

  function updateField(field, value) {
    setProposal(prev => ({ ...prev, [field]: field === 'use' ? value : Number(value) }));
  }

  async function submit(event) {
    event.preventDefault();
    const evaluation = await evaluate(zone, proposal);
    setResult(evaluation);
  }

  return (
    <main className="container">
      <h1>City Zoning Review</h1>
      <p className="subtitle">Check a building proposal against municipal zoning rules.</p>
      <form onSubmit={submit} className="grid">
        <label>
          Zone
          <select value={zone} onChange={e => setZone(e.target.value)}>
            {Object.entries(zones).map(([code, z]) => (
              <option key={code} value={code}>{code} - {z.name}</option>
            ))}
          </select>
        </label>
        <label>
          Use
          <input value={proposal.use} onChange={e => updateField('use', e.target.value)} />
        </label>
        {NUMERIC_FIELDS.map(field => (
          <label key={field}>
            {field}
            <input type="number" value={proposal[field]} onChange={e => updateField(field, e.target.value)} />
          </label>
        ))}
        <button type="submit">Evaluate</button>
      </form>
      {result && (
        <section className={result.compliant ? 'result ok' : 'result bad'}>
          <h2>{result.compliant ? 'Compliant' : 'Not Compliant'}</h2>
          <p>Zone: {result.zone}</p>
          <p>Permit fee: ${result.permitFee}</p>
          {result.violations && result.violations.length > 0 && (
            <ul>
              {result.violations.map((v, i) => <li key={i}>{v.rule}: {v.message}</li>)}
            </ul>
          )}
        </section>
      )}
    </main>
  );
}
