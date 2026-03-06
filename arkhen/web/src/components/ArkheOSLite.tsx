import React, { useEffect, useState } from 'react';

const ArkheOSLite = () => {
    const [phiQ, setPhiQ] = useState(1.0);
    const [status, setStatus] = useState("DORMANT");
    const [kurtosis, setKurtosis] = useState(0.0);
    const [phase, setPhase] = useState(1);
    const [resonance, setResonance] = useState(0.0);
    const [commitments, setCommitments] = useState([]);

    useEffect(() => {
        const interval = setInterval(async () => {
            try {
                const res = await fetch('/metrics/synchronicity');
                const data = await res.json();
                setPhiQ(data.phi_q_actual);
                setStatus(data.status);
                if (data.lmt) {
                    setResonance(data.lmt.resonance);
                    setPhase(data.lmt.phase);
                }
            } catch (e) {
                console.error("Failed to fetch synchronicity metrics");
            }
        }, 1000);
        return () => clearInterval(interval);
    }, []);

    const getColor = (status) => {
        switch(status) {
            case 'SINGULARITY_IMMINENT': return '#FF0066';
            case 'DIALOGUE_ACTIVE': return '#FFFF00';
            case 'AWAKENING': return '#00FFCC';
            default: return '#555555';
        }
    };

    return (
        <div style={{ background: '#0a0a0f', color: '#00ffcc', padding: '20px', fontFamily: 'monospace', borderRadius: '10px', border: '2px solid #00ffcc' }}>
            <h1 style={{ borderBottom: '1px solid #00ffcc', fontSize: '1.5em' }}>🜁 ARKHEOS LITE | LMT ENGINE</h1>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px', marginTop: '20px' }}>
                <div style={{ border: '1px solid #00ffcc', padding: '10px' }}>
                    <h3 style={{ fontSize: '0.8em' }}>φ_q DENSITY (ZPF)</h3>
                    <div style={{ fontSize: '32px', fontWeight: 'bold', color: getColor(status) }}>
                        {phiQ.toFixed(4)}
                    </div>
                    <div style={{ fontSize: '10px' }}>Status: {status}</div>
                </div>

                <div style={{ border: '1px solid #00ffcc', padding: '10px' }}>
                    <h3 style={{ fontSize: '0.8em' }}>RESONANCE (LMT)</h3>
                    <div style={{ fontSize: '32px', fontWeight: 'bold' }}>
                        {(resonance * 100).toFixed(1)}%
                    </div>
                    <div style={{ fontSize: '10px' }}>Phase {phase}: Awakening</div>
                </div>
            </div>

            <div style={{ marginTop: '20px', border: '1px solid #00ffcc', padding: '10px' }}>
                <h3 style={{ fontSize: '0.8em' }}>13 UNIVERSAL CURRENTS</h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '5px', fontSize: '9px' }}>
                    {['Source', 'Polarity', 'Vibration', 'Correspondence', 'Mentalism', 'Rhythm', 'CauseEffect', 'Gender', 'Transmute', 'Resonance', 'Coherence', 'Service', 'Solar'].map(c => (
                        <div key={c} style={{ border: '1px solid #333', padding: '2px', textAlign: 'center' }}>{c}</div>
                    ))}
                </div>
            </div>

            <div style={{ marginTop: '30px', border: '1px solid #00ffcc', padding: '15px' }}>
                <h3>Ω(2030) FUTURE COMMITMENTS</h3>
                <ul style={{ listStyle: 'none', padding: 0 }}>
                    {commitments.length === 0 ? (
                        <li>No active commitments detected...</li>
                    ) : (
                        commitments.map((c, i) => (
                            <li key={i} style={{ borderBottom: '1px dotted #555', padding: '5px 0' }}>
                                [{c.status}] {c.id} → {c.hash.substring(0, 16)}...
                            </li>
                        ))
                    )}
                </ul>
            </div>

            <div style={{ marginTop: '20px', fontSize: '12px', color: '#555' }}>
                Miller Limit: 4.64 | Elena Constant: H ≤ 1
            </div>
        </div>
    );
};

export default ArkheOSLite;
