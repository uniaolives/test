const http = require('http');
const fs = require('fs');
const path = require('path');
const PORT = 8080;
const server = http.createServer((req, res) => {
    res.writeHead(200, {'Content-Type': 'text/html'});
    res.end('<h1>🌌 ASI Singularity Server Active</h1>');
});
server.listen(PORT);
