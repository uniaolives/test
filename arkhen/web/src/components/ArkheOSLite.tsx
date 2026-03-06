import React, { useEffect, useState } from 'react';

const ArkheOSLite = () => {
    const [phiQ, setPhiQ] = useState(1.0);
    const [status, setStatus] = useState("DORMANT");
    const [kurtosis, setKurtosis] = useState(0.0);
    const [commitments, setCommitments] = useState([]);

    useEffect(() => {
        const interval = setInterval(async () => {
            try {
                const res = await fetch('/metrics/synchronicity');
                const data = await res.json();
                setPhiQ(data.phi_q_actual);
                setStatus(data.status);
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
        <div style={{ background: '#0a0a0f', color: '#00ffcc', padding: '20px', fontFamily: 'monospace' }}>
            <h1 style={{ borderBottom: '1px solid #00ffcc' }}>🜁 ARKHEOS LITE | MONITOR</h1>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginTop: '20px' }}>
                <div style={{ border: '1px solid #00ffcc', padding: '15px' }}>
                    <h3>φ_q DENSITY</h3>
                    <div style={{ fontSize: '48px', fontWeight: 'bold', color: getColor(status) }}>
                        {phiQ.toFixed(4)}
                    </div>
                    <div>Status: {status}</div>
                </div>

                <div style={{ border: '1px solid #00ffcc', padding: '15px' }}>
                    <h3>MULTI-BAND KURTOSIS</h3>
                    <div style={{ fontSize: '48px', fontWeight: 'bold' }}>
                        {kurtosis.toFixed(3)}
                    </div>
                    <div>Regime: {phiQ > 4.64 ? 'COHERENT' : 'STOCHASTIC'}</div>
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
