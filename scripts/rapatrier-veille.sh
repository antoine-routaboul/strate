#!/usr/bin/env bash
# Rapatrie sur le Mac la veille qui tourne sur une machine distante (voir docs/deploiement-azure.md).
#
#   scripts/rapatrier-veille.sh strate-vm          # nom d'hôte de ~/.ssh/config
#
# 1. copie cohérente de la base sur la machine (`strate veille backup`, jamais un cp pendant l'écriture) ;
# 2. téléchargement, puis remplacement de data/veille/veille.db. Les fichiers -wal et -shm de l'ancienne
#    base sont retirés d'abord : laissés là, SQLite les rejouerait sur la nouvelle base et l'abîmerait ;
# 3. signaux satellite calculés sur la machine (data/signals), ajoutés à ceux du Mac sans écraser un
#    fichier plus récent du Mac (rsync -u).
#
# Arrêter `strate serve` avant (il garde la base ouverte) et ne pas faire tourner la veille sur le Mac en
# même temps.
set -euo pipefail

host="${1:?usage : scripts/rapatrier-veille.sh <hôte ssh> [dossier distant, défaut strate]}"
remote_dir="${2:-strate}"
here="$(cd "$(dirname "$0")/.." && pwd)"
data="${STRATE_DATA:-$here/data}"

if pgrep -f "strate serve" >/dev/null 2>&1; then
  echo "strate serve tourne sur ce Mac : l'arrêter avant de remplacer la base" >&2
  exit 1
fi
if pgrep -f "strate veille run" >/dev/null 2>&1; then
  echo "la veille tourne aussi sur ce Mac : uv run strate veille uninstall" >&2
  exit 1
fi

# dossier temporaire sur le même disque que la base : le mv final est un simple renommage
mkdir -p "$data/veille"
tmp="$(mktemp -d "$data/veille/.rapatriement.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT

# uv par son chemin complet : une commande ssh non interactive ne lit pas le PATH de la session
ssh "$host" "cd $(printf %q "$remote_dir") && ~/.local/bin/uv run strate veille backup --to /tmp/veille-rapatriee.db"
scp -q "$host:/tmp/veille-rapatriee.db" "$tmp/veille.db"
ssh "$host" "rm -f /tmp/veille-rapatriee.db"

rm -f "$data/veille/veille.db-wal" "$data/veille/veille.db-shm"
mv "$tmp/veille.db" "$data/veille/veille.db"
echo "base remplacée : $data/veille/veille.db"

mkdir -p "$data/signals"
rsync -au "$host:$remote_dir/data/signals/" "$data/signals/"
echo "signaux satellite à jour : $data/signals"
