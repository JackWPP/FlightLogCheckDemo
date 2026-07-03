#!/usr/bin/env bash
# Flight Log Check Demo — server-side helper.
# Wraps the most common operations so you don't have to remember docker compose incantations.
#
# Usage:
#   ./deploy/deploy.sh up        # build (if needed) + start in background
#   ./deploy/deploy.sh down      # stop
#   ./deploy/deploy.sh restart   # recreate app container (re-read .env)
#   ./deploy/deploy.sh update    # git pull + rebuild + restart
#   ./deploy/deploy.sh logs      # tail logs (Ctrl-C to exit)
#   ./deploy/deploy.sh status    # show container status + health
#   ./deploy/deploy.sh shell     # bash into the app container
#   ./deploy/deploy.sh backup    # tar up out/ and outputs/

set -euo pipefail

cd "$(dirname "$0")/.."

cmd="${1:-help}"

ensure_env() {
    if [[ ! -f .env ]]; then
        echo "[deploy] .env not found, copying from .env.example"
        cp .env.example .env
        echo "[deploy] >>> Edit .env and fill in your API keys before going live!"
        echo "[deploy] >>> Continuing anyway — demo cache will still work."
    fi
}

case "$cmd" in
    up)
        ensure_env
        docker compose build
        docker compose up -d
        docker compose ps
        ;;
    down)
        docker compose down
        ;;
    restart)
        docker compose up -d --force-recreate app
        docker compose ps
        ;;
    update)
        ensure_env
        git pull --ff-only
        docker compose build
        docker compose up -d
        docker compose ps
        ;;
    logs)
        docker compose logs -f --tail=200
        ;;
    status)
        docker compose ps
        echo
        echo "[deploy] Disk usage:"
        du -sh out outputs 2>/dev/null || true
        ;;
    shell)
        docker compose exec app bash
        ;;
    backup)
        ts="$(date +%Y%m%d-%H%M%S)"
        out="backup-${ts}.tgz"
        tar -czf "$out" out outputs 2>/dev/null || true
        echo "[deploy] wrote $out ($(du -h "$out" | cut -f1))"
        ;;
    help|*)
        sed -n '2,20p' "$0"
        ;;
esac