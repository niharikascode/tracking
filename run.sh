#!/usr/bin/env bash
# Start the SAM 3 tracker web app.
#   ./run.sh            -> http://localhost:7860 (this machine only)
#   ./run.sh --public   -> also opens a Cloudflare quick tunnel and prints a public https URL
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-7860}"
CLOUDFLARED="${CLOUDFLARED:-$(command -v cloudflared || echo "$HOME/bin/cloudflared")}"

if [[ ! -f .env ]] || ! grep -q '^HF_TOKEN=.\+' .env; then
  echo "error: put HF_TOKEN=... in $(pwd)/.env (see .env.example)" >&2
  exit 1
fi

if [[ "${1:-}" == "--public" ]] && ! grep -q '^APP_PASSWORD=.\+' .env; then
  echo "warning: no APP_PASSWORD in .env, so anyone with the public URL can use your GPU." >&2
  echo "         add APP_PASSWORD=<something> to .env to require an access key." >&2
fi

.venv/bin/uvicorn server:app --host 127.0.0.1 --port "$PORT" &
SERVER_PID=$!
trap 'kill $SERVER_PID ${TUNNEL_PID:-} 2>/dev/null' EXIT

echo "local:  http://localhost:$PORT"

if [[ "${1:-}" == "--public" ]]; then
  "$CLOUDFLARED" tunnel --no-autoupdate --url "http://127.0.0.1:$PORT" 2>&1 \
    | tee tunnel.log | grep --line-buffered -oE 'https://[a-z0-9-]+\.trycloudflare\.com' \
    | sed -u 's/^/public: /' &
  TUNNEL_PID=$!
fi

wait $SERVER_PID
