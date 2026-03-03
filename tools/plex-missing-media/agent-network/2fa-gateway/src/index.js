import express from 'express';
import TelegramBot from 'node-telegram-bot-api';
import crypto from 'crypto';
import dotenv from 'dotenv';

dotenv.config();

const app = express();
app.use(express.json());

const bot = new TelegramBot(process.env.TELEGRAM_BOT_TOKEN);
const ownerChatId = process.env.TELEGRAM_CHAT_ID;
const pendingApprovals = new Map();
const PROXY_HMAC_SECRET = process.env.PROXY_HMAC_SECRET;

function verifyProxyHMAC(req) {
  const signature = req.headers['x-proxy-signature'];
  const timestamp = req.headers['x-proxy-timestamp'];
  const body = JSON.stringify(req.body);
  const message = `${req.method}:${req.path}:${timestamp}:${body}`;
  const hmac = crypto.createHmac('sha256', PROXY_HMAC_SECRET).update(message).digest('hex');
  return crypto.timingSafeEqual(Buffer.from(signature), Buffer.from(hmac));
}

app.post('/request-approval', (req, res) => {
  if (!verifyProxyHMAC(req)) {
    return res.status(401).json({ error: 'Invalid HMAC' });
  }

  const { operationId, description, metadata } = req.body;
  const baseUrl = process.env.PUBLIC_URL || `http://localhost:${process.env.PORT || 4000}`;
  const approveUrl = `${baseUrl}/approve/${operationId}`;
  const rejectUrl = `${baseUrl}/reject/${operationId}`;

  const message = `
🔐 *Aprovação de Operação Requerida*

*Agente:* Plex Missing Media v5.1
*Operação:* ${description}
*ID:* \`${operationId}\`
*Timestamp:* ${new Date().toISOString()}

[✅ Aprovar](${approveUrl}) | [❌ Rejeitar](${rejectUrl})

_Esta solicitação expira em 5 minutos._
  `;

  bot.sendMessage(ownerChatId, message, { parse_mode: 'Markdown' })
    .catch(err => console.error('Telegram send error:', err));

  pendingApprovals.set(operationId, {
    approved: null,
    createdAt: Date.now(),
    metadata
  });

  res.json({ status: 'pending', operationId });
});

app.get('/approve/:id', (req, res) => {
  const op = pendingApprovals.get(req.params.id);
  if (op) {
    op.approved = true;
    res.send(`<html><body style="background:#0a0a0a; color:#0f0; display:flex; justify-content:center; align-items:center; height:100vh;"><h1>✅ Operação Aprovada</h1></body></html>`);
  } else {
    res.status(404).send('Operação não encontrada.');
  }
});

app.get('/reject/:id', (req, res) => {
  const op = pendingApprovals.get(req.params.id);
  if (op) {
    op.approved = false;
    res.send(`<html><body style="background:#0a0a0a; color:#f00; display:flex; justify-content:center; align-items:center; height:100vh;"><h1>❌ Operação Rejeitada</h1></body></html>`);
  } else {
    res.status(404).send('Operação não encontrada.');
  }
});

app.get('/approval-status/:id', (req, res) => {
  const op = pendingApprovals.get(req.params.id);
  if (!op) return res.status(404).json({ error: 'Not found' });
  if (Date.now() - op.createdAt > 5 * 60 * 1000) {
    pendingApprovals.delete(req.params.id);
    return res.json({ approved: false, reason: 'timeout' });
  }
  if (op.approved === null) return res.json({ approved: null, status: 'pending' });
  pendingApprovals.delete(req.params.id);
  res.json({ approved: op.approved });
});

const PORT = process.env.PORT || 4000;
app.listen(PORT, () => console.log(`📱 2FA Gateway ativo na porta ${PORT}`));
