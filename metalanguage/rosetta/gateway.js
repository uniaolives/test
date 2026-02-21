const express = require('express');
const app = express();

app.get('/web2/ingest', async (req, res) => {
    const humanData = await fetchExternalAPI();
    const cleanData = sanitizeForOntologicalGateway(humanData);
    hypergraphQueue.push(cleanData);
    res.send({ status: "Data digested by ProtoAGI" });
});
