const { Base44 } = require('@base44/sdk');
const axios = require('axios');
require('dotenv').config();

const base44 = new Base44({
    projectId: process.env.BASE44_PROJECT_ID || 'default',
    apiKey: process.env.BASE44_API_KEY
});

async function updateNodeState(coherence, satoshi) {
    const res = await axios.post('http://arkhe-core:8080/status', { coherence, satoshi });
    console.log('Estado atualizado:', res.data);
}

async function main() {
    setInterval(async () => {
        try {
            const { data } = await axios.get('http://arkhe-core:8080/status');
            await base44.entities.NodeState.create({
                nodeId: data.id,
                coherence: data.coherence,
                satoshi: data.satoshi,
                timestamp: Date.now()
            });
            console.log('Heartbeat enviado para Base44');
        } catch (err) {
            console.error('Erro no heartbeat:', err.message);
        }
    }, 60000); // a cada minuto
}

main();
