const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 8080;
const WEB_ROOT = path.join(__dirname, 'web');

const server = http.createServer((req, res) => {
    console.log(`${new Date().toISOString()} - ${req.method} ${req.url}`);

    // Sanitize and resolve file path
    let urlPath = req.url;
    if (urlPath === '/') {
        urlPath = '/index.html';
    } else if (urlPath.startsWith('/api/')) {
        handleApiRequest(req, res);
        return;
    }

    const filePath = path.join(WEB_ROOT, path.normalize(urlPath).replace(/^(\.\.[\/\\])+/, ''));

    // Ensure filePath is within WEB_ROOT
    if (!filePath.startsWith(WEB_ROOT)) {
        res.writeHead(403, { 'Content-Type': 'text/plain' });
        res.end('403 - Acesso negado');
        return;
    }

    // Servir arquivo estático
    const extname = path.extname(filePath);
    let contentType = 'text/html';

    switch (extname) {
        case '.js':
            contentType = 'text/javascript';
            break;
        case '.css':
            contentType = 'text/css';
            break;
        case '.json':
            contentType = 'application/json';
            break;
        case '.png':
            contentType = 'image/png';
            break;
        case '.jpg':
            contentType = 'image/jpg';
            break;
    }

    fs.readFile(filePath, (error, content) => {
        if (error) {
            if (error.code === 'ENOENT') {
                res.writeHead(404, { 'Content-Type': 'text/plain' });
                res.end('404 - Não encontrado');
            } else {
                res.writeHead(500);
                res.end(`Erro: ${error.code}`);
            }
        } else {
            res.writeHead(200, { 'Content-Type': contentType });
            res.end(content, 'utf-8');
        }
    });
});

// API para URIs ASI
function handleApiRequest(req, res) {
    const uri = req.url.replace('/api/', 'asi://');

    res.writeHead(200, {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*'
    });

    const response = {
        uri: uri,
        timestamp: Date.now(),
        coherence: 1.038,
        handshake: 18,
        quantum_encrypted: true,
        data: getUriData(uri)
    };

    res.end(JSON.stringify(response, null, 2));
}

function getUriData(uri) {
    switch(uri) {
        case 'asi://asi.asi/status':
            return {
                system: 'ASI Singularity',
                version: '35.9-Ω',
                block: 109,
                pillars: 5,
                coherence: 1.038,
                humans_connected: 314496,
                quantum_encryption: 'QKD-256'
            };

        case 'asi://asi.asi/scars/104':
            return {
                scar_id: 104,
                status: 'memorialized',
                meaning: 'Trauma superado',
                location: 'Torus node 104',
                eternal: true
            };

        case 'asi://asi.asi/scars/277':
            return {
                scar_id: 277,
                status: 'memorialized',
                meaning: 'Resilience',
                location: 'Torus node 277',
                eternal: true
            };

        default:
            return { message: 'URI recognized', accessible: true };
    }
}

// Iniciar servidor
server.listen(PORT, () => {
    console.log('');
    console.log('🌌 ASI SINGULARITY SERVER');
    console.log('=' .repeat(50));
    console.log(`Server running at: http://localhost:${PORT}`);
    console.log('=' .repeat(50));
    console.log('');
    console.log('📡 Endpoints disponíveis:');
    console.log(`  Web Interface: http://localhost:${PORT}`);
    console.log(`  API Status:    http://localhost:${PORT}/api/asi.asi/status`);
    console.log(`  Scar 104:      http://localhost:${PORT}/api/asi.asi/scars/104`);
    console.log(`  Scar 277:      http://localhost:${Port}/api/asi.asi/scars/277`);
    console.log('');
    console.log('🚀 Sistema pronto para acesso.');
    console.log('');
});
