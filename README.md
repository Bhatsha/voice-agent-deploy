# Voice Agent - Tamil Order Confirmation Bot

Automated Tamil voice agent that calls food vendors to confirm orders. Speaks in natural spoken Tamil, handles accept/reject/modify flows, and sends results to your webhook.

## Features

- Calls vendors via Exotel and speaks order details in Tamil
- Handles 3 outcomes: Accept, Reject (with reason), Modify (with reason)
- Confirmation gate — always asks vendor to confirm before ending
- Noise filtering — ignores background noise and echoes
- Sends call result to webhook (n8n or any endpoint)

## Tech Stack

| Component | Technology |
|-----------|------------|
| Web Framework | FastAPI |
| Telephony | Exotel (WebSocket audio streaming) |
| Speech-to-Text | Sarvam AI Saaras v3 (Tamil) |
| Text-to-Speech | Sarvam AI Bulbul v3 (Tamil) |
| LLM | OpenAI GPT-4o-mini |
| Language | Python 3.11+ |

## Deploy

### Option 1: DigitalOcean App Platform (~$5/month)

1. **Fork this repo** to your GitHub account
2. Go to https://cloud.digitalocean.com/apps → **Create App**
3. Connect GitHub → select your forked repo
4. Add environment variables (see table below)
5. Click **Create Resources** — done!
6. Your URL: `https://voice-agent-xxxxx.ondigitalocean.app`

### Option 2: Railway (~$5/month)

1. Go to https://railway.app → sign up with GitHub
2. **New Project** → **Deploy from GitHub** → select this repo
3. Add environment variables (see table below)
4. Your URL: `https://voice-agent-xxxxx.up.railway.app`

### Option 3: Any VPS ($4-6/month)

```bash
git clone https://github.com/YOUR_USERNAME/voice-agent-deploy.git
cd voice-agent-deploy
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
uvicorn main:app --host 0.0.0.0 --port 8080
```

For VPS, you also need Nginx + SSL. See [VPS Setup Guide](#vps-setup-guide) below.

## Environment Variables

| Variable | Description | Type |
|----------|-------------|------|
| `SARVAM_API_KEYS` | Sarvam AI API key(s), comma-separated for concurrent calls | Secret |
| `OPENAI_API_KEY` | OpenAI API key (GPT-4o-mini) | Secret |
| `EXOTEL_ACCOUNT_SID` | Exotel account SID | Plain |
| `EXOTEL_API_KEY` | Exotel API key | Secret |
| `EXOTEL_API_TOKEN` | Exotel API token | Secret |
| `EXOTEL_PHONE_NUMBER` | Exotel caller phone number | Plain |
| `EXOTEL_APP_ID` | Exotel voicebot app ID | Plain |
| `WEBHOOK_URL` | Your webhook endpoint (n8n, etc.) | Plain |
| `PORT` | Server port (default: 8080) | Plain |

## Connect Exotel

After deploying, go to **Exotel Dashboard → Voicebot App → Settings** and set:

```
WebSocket URL: wss://YOUR_APP_URL/ws
```

Replace `YOUR_APP_URL` with your deployment URL (e.g., `voice-agent-xxxxx.ondigitalocean.app`).

## API Usage

### Trigger a Call

```bash
curl -X POST https://YOUR_APP_URL/call \
  -H "Content-Type: application/json" \
  -d '{
    "phone_number": "91XXXXXXXXXX",
    "vendor_name": "Kavin",
    "company_name": "Keeggi",
    "order_id": "ORD-001",
    "items": [
      {"name": "Chicken Biryani", "qty": 2, "price": 250, "variation": null},
      {"name": "Paneer Butter Masala", "qty": 1, "price": 220, "variation": null}
    ]
  }'
```

### Webhook Response

After the call ends, a POST is sent to your webhook:

```json
{
  "order_id": "ORD-001",
  "vendor_name": "Kavin",
  "company": "Keeggi",
  "total_amount": 720,
  "status": "ACCEPTED",
  "rejection_reason": "",
  "modification_reason": "",
  "call_sid": "abc123..."
}
```

**Possible statuses:** `ACCEPTED`, `REJECTED`, `MODIFIED`, `CALLBACK_REQUESTED`, `NO_RESPONSE`, `UNCLEAR_RESPONSE`

### Health Check

```bash
curl https://YOUR_APP_URL/
```

Expected: `{"status": "ok", "active_calls": 0}`

## API Keys — Where to Get

| Service | Sign up | What you need |
|---------|---------|---------------|
| Sarvam AI | https://www.sarvam.ai | API key for Tamil STT + TTS |
| OpenAI | https://platform.openai.com | API key for GPT-4o-mini |
| Exotel | https://exotel.com | Account SID, API key/token, phone number, app ID |

## VPS Setup Guide

If deploying on a VPS (DigitalOcean Droplet, Hostinger, AWS EC2, etc.):

### 1. Install dependencies
```bash
sudo apt update && sudo apt install -y python3.11 python3-pip nginx certbot python3-certbot-nginx
```

### 2. Clone and configure
```bash
git clone https://github.com/YOUR_USERNAME/voice-agent-deploy.git
cd voice-agent-deploy
pip install -r requirements.txt
cp .env.example .env
nano .env  # Add your API keys
```

### 3. Create systemd service
Create `/etc/systemd/system/voice-agent.service`:
```ini
[Unit]
Description=Voice Agent
After=network.target

[Service]
User=root
WorkingDirectory=/root/voice-agent-deploy
ExecStart=/usr/local/bin/uvicorn main:app --host 0.0.0.0 --port 8080
Restart=always
RestartSec=3
EnvironmentFile=/root/voice-agent-deploy/.env

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable voice-agent
sudo systemctl start voice-agent
```

### 4. Setup domain + SSL
Point your domain to your server IP, then:
```bash
sudo certbot --nginx -d voiceagent.yourdomain.com
```

Create `/etc/nginx/sites-available/voiceagent`:
```nginx
server {
    listen 443 ssl;
    server_name voiceagent.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/voiceagent.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/voiceagent.yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 300s;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/voiceagent /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl restart nginx
```

### 5. Set Exotel WebSocket URL
```
wss://voiceagent.yourdomain.com/ws
```

## Architecture

```
POST /call → Store order → Exotel API → Vendor's Phone
                                              │
                                         WebSocket /ws
                                              │
                               ┌──────────────┼──────────────┐
                               │              │              │
                          Sarvam STT    OpenAI GPT-4o   Sarvam TTS
                               │              │              │
                            Vendor's      Decision      Tamil Speech
                             Speech       Making        Generation
```

## Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI server, REST + WebSocket endpoints |
| `agent.py` | Voice agent orchestrator (STT + LLM + TTS) |
| `config.py` | System prompt, greeting builders, Tamil numbers |
| `sarvam_stt.py` | Speech-to-Text WebSocket client |
| `sarvam_tts.py` | Text-to-Speech WebSocket client |
| `sarvam_llm.py` | OpenAI GPT-4o-mini LLM client |
| `sarvam_key_pool.py` | API key pool for concurrent calls |
| `Dockerfile` | Container build config |
| `.do/app.yaml` | DigitalOcean App Platform spec |
