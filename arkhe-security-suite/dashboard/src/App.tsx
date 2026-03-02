import React from 'react';
import PhiKnob from './components/PhiKnob';
import ThreatMap from './components/ThreatMap';
import EntanglementGraph from './components/EntanglementGraph';

function App() {
  return (
    <div className="App">
      <h1>Arkhe(n) Security Suite (Ω+209)</h1>
      <PhiKnob />
      <ThreatMap />
      <EntanglementGraph />
    </div>
  );
}

export default App;
