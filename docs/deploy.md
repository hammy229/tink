# Deploying Tink

## Free options first

**1. Hugging Face Space (free, always-on link, recommended for Devpost).** 2 vCPU and 16 GB RAM with HTTPS, built from our Dockerfile.

```bash
uvx --from huggingface_hub hf auth login     # once; paste a "Write" token from huggingface.co/settings/tokens
XAI_API_KEY=... uv run python deploy/hf_space.py   # key optional; becomes a Space secret
```

It prints the Space page (build logs) and the app URL, `https://<user>-tink.hf.space`. The first build takes about 10 minutes. Recordings reset when the Space restarts, and it sleeps after about 48 hours without visitors, waking on the next visit.

**2. Cloudflare quick tunnel (free, no account, runs from your laptop).** Good as a judging-day backup.

```bash
TINK_DATA=data/public-demo TINK_PUBLIC=1 uv run uvicorn tink.app:app --port 8768   # a separate public copy
cloudflared tunnel --url http://localhost:8768                                    # prints https://....trycloudflare.com
```

Point the tunnel at the public copy, never at your own local Tink: anyone with the link can use whatever it serves. The link dies when the laptop sleeps or the command stops.

## Paid option: your own VM

The rest of this guide sets up a VM (Vultr or similar) with Caddy and a real domain. It costs money; use it only if you have credits.


Tink needs HTTPS. Browsers block the microphone on plain HTTP.

## A. Vultr VM (Docker + Caddy)

1. Create a Vultr Cloud Compute VM: Ubuntu 24.04, at least 2 vCPU and 4 GB RAM (the model loads into memory).
2. SSH in and install Docker: `curl -fsSL https://get.docker.com | sh`
3. Clone the repo: `git clone <your-repo-url> tink && cd tink`
4. Create `.env`:
   ```
   TINK_DOMAIN=yourname.tech
   XAI_API_KEY=your-key   # optional
   ```
5. Point your .tech domain at the VM: add an A record for the domain with the VM's public IP. Wait for DNS to update.
   No domain yet? Use `TINK_DOMAIN=<server-ip>.sslip.io` (for example `203.0.113.7.sslip.io`).
6. Open ports 80 and 443 in the Vultr firewall and `ufw` if enabled.
7. Start it: `docker compose up -d --build`. The first build takes several minutes (it downloads the model).
8. Logs: `docker compose logs -f`. Update later with `git pull && docker compose up -d --build`.

Check it worked: `curl https://DOMAIN/api/sounds` should return JSON.

## B. Hugging Face Spaces (free fallback)

1. Create a new Space, SDK = Docker, and push this repo to it.
2. Add `app_port: 7860` to the Space README front matter, and in Settings add the variable `PORT=7860` so uvicorn listens there.
3. Set the variable `TINK_PUBLIC=1` and the secret `XAI_API_KEY`.
4. Spaces serves HTTPS for you, so the microphone works.
5. Storage is ephemeral: recorded data is lost on restart or rebuild. Free CPU hardware is slow on the first request.

Check it worked: `curl https://<space-subdomain>.hf.space/api/sounds`.

## Troubleshooting

- "Microphone blocked" means the page is not on HTTPS. Use the domain (or sslip.io name), not the bare IP or http://.
- Certificate errors: confirm DNS points at the VM and ports 80/443 are open, then `docker compose logs caddy`.
