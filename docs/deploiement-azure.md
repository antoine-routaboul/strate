# Faire tourner la veille 24 h/24 sur une machine Azure

La veille (`strate veille run`) enchaîne collecte, vérification et corrélations toutes les 15 minutes. Sur un
portable, elle s'arrête dès que le Mac dort. Ce guide la place sur une petite machine virtuelle Linux chez
Microsoft Azure. Le Mac garde l'interface et récupère la base quand on veut.

Durée : une heure environ la première fois. Coût visé : zéro pendant 12 mois, dans les offres gratuites,
à vérifier dans le portail (voir § 8).

## 1. Ce qu'il faut

- Un compte Azure.
  - **Azure for Students** : 100 $ de crédit sur 12 mois, sans carte bancaire, avec une adresse d'école.
    Sans carte, aucune facture surprise n'est possible : à la fin du crédit, l'abonnement est suspendu.
  - Sinon, **le compte gratuit** : 200 $ à utiliser en 30 jours, carte demandée mais non débitée. Pour
    garder les 12 mois gratuits au-delà des 30 jours, il faut passer au paiement à l'usage : la limite de
    dépenses disparaît alors, et l'alerte de budget (§ 8) devient le seul garde-fou.

  Les deux offres comprennent 750 h par mois de petites machines (B1s, B2ats v2, B2pts v2) pendant
  12 mois, soit une machine allumée en permanence.
- Sur le Mac : le dépôt Strate avec ses données (`data/catalog/mines.parquet` existe, sinon lancer d'abord
  `uv run strate catalog`).

## 2. Créer la machine (portail Azure)

Portail → **Services gratuits** (*Free services*) → **Machine virtuelle Linux** → **Créer** : une machine
créée ailleurs n'est pas toujours rattachée à l'offre gratuite. À défaut, **Machines virtuelles** →
**Créer** avec les réglages ci-dessous, puis vérifier dans *Analyse des coûts*. Les réglages qui comptent :

| Réglage | Valeur | Pourquoi |
|---|---|---|
| Région | **West Europe** | les images Sentinel de Planetary Computer y sont stockées : lecture rapide et gratuite |
| Image | **Ubuntu Server 24.04 LTS, x64 Gen2** | les paquets Python s'y installent sans compilation |
| Taille | **B2ats v2** (2 vCPU, 1 Gio) | dans l'offre gratuite. B1s l'est aussi mais manque souvent de stock |
| Authentification | **Clé publique SSH**, utilisateur `azureuser` | pas de mot de passe exposé sur Internet |
| Ports entrants publics | **SSH (22) seulement** | l'interface reste sur le Mac, rien d'autre à ouvrir |
| Disque du système | **SSD Premium, 64 Gio** | le cache d'images satellite grossit (§ 7). L'offre gratuite inclut, selon Microsoft, deux disques P6 de 64 Gio : à vérifier dans *Analyse des coûts* après 48 h |

Laisser le reste par défaut. Avec une nouvelle clé, le portail télécharge un fichier `.pem` : le garder, c'est
la seule clé d'accès à la machine.

Noter l'**adresse IP publique** affichée à la fin.

## 3. Se connecter depuis le Mac

Ranger la clé et créer un raccourci `strate-vm` :

```bash
mkdir -p ~/.ssh && mv ~/Downloads/<nom-de-la-clé>.pem ~/.ssh/strate-vm.pem && chmod 400 ~/.ssh/strate-vm.pem
```

Ajouter à `~/.ssh/config` (remplacer l'adresse) :

```
Host strate-vm
    HostName <adresse-IP>
    User azureuser
    IdentityFile ~/.ssh/strate-vm.pem
```

Puis :

```bash
ssh strate-vm
```

## 4. Préparer la machine (dans la session SSH)

Mémoire d'échange de 2 Gio : la machine n'a que 1 Gio de mémoire vive. La veille en prend environ 360 Mo
(index des 80 000 noms de mines), une analyse satellite environ 300 Mo de plus (mesuré sur Mac).

```bash
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

Outils, uv et le code :

```bash
sudo apt-get update && sudo apt-get install -y git rsync sqlite3   # sqlite3 : facultatif, pour inspecter
curl -LsSf https://astral.sh/uv/install.sh | sh
exec bash -l    # recharge le PATH : uv est dans ~/.local/bin
git clone https://github.com/antoine-routaboul/strate.git ~/strate
cd ~/strate && uv sync --locked
```

## 5. Copier les données depuis le Mac

`data/` n'est pas dans le dépôt (sources tierces). La veille n'a besoin que de quatre éléments, environ 35 Mo.
Depuis le dossier du projet **sur le Mac** :

```bash
ssh strate-vm 'mkdir -p ~/strate/data/catalog ~/strate/data/sources/maus2022 ~/strate/data/cache'
rsync -a data/catalog/mines.parquet strate-vm:strate/data/catalog/
rsync -a data/sources/maus2022/global_mining_polygons_v2.gpkg strate-vm:strate/data/sources/maus2022/
rsync -a data/signals strate-vm:strate/data/
rsync -a data/cache/radar strate-vm:strate/data/cache/
```

| Élément | Sans lui |
|---|---|
| `catalog/mines.parquet` (catalogue) | la veille ne reconnaît que les quelques mines suivies de `sites.toml`, sans erreur visible |
| `maus2022/…gpkg` (emprises des mines) | chaque analyse satellite échoue à l'étape radar |
| `signals/` (analyses déjà faites) | les mines déjà analysées sur le Mac sont retéléchargées depuis 2019 |
| `cache/radar/` | quelques téléchargements radar refaits |

La base de veille du Mac n'est pas copiée : la machine repart d'une base neuve.

## 6. Lancer la veille

Si la veille tourne aussi en service sur le Mac, l'y arrêter d'abord (dans le dossier du projet, sur le
Mac) : `uv run strate veille uninstall`. Deux veilles en parallèle écriraient deux bases différentes.

Dans la session SSH, un premier cycle à la main pour voir que tout marche. Il commence à rattraper les
48 dernières heures de GDELT (40 fichiers par cycle, la suite aux cycles suivants) ; avec des analyses
satellite, compter jusqu'à une demi-heure :

```bash
cd ~/strate && uv run strate veille once
```

Chaque étape doit sortir `"ok": true`. Puis le service permanent :

```bash
uv run strate veille install
```

Il écrit `~/.config/systemd/user/strate-veille.service`, l'active et active le « linger ». Sans linger,
systemd arrêterait la veille à la déconnexion SSH. Si la commande affiche `attention : linger refusé`,
lancer `sudo loginctl enable-linger "$USER"`.

Vérifier :

```bash
uv run strate veille status          # étapes, totaux, « service (base) : actif »
systemctl --user status strate-veille
loginctl show-user "$USER" -p Linger  # Linger=yes
tail -f logs/veille.log              # Ctrl-C pour quitter
```

Se déconnecter (`exit`) : la veille continue. Elle repart seule après un redémarrage de la machine.

## 7. Au quotidien

**Récupérer la veille sur le Mac.** Arrêter `strate serve`, puis depuis le dossier du projet :

```bash
scripts/rapatrier-veille.sh strate-vm
```

Le script fait une copie cohérente de la base sur la machine (`strate veille backup`), la télécharge,
remplace `data/veille/veille.db` et ajoute les signaux satellite calculés là-bas (sans écraser un signal
plus récent du Mac). Relancer ensuite `uv run strate serve` : l'interface montre les articles et les
corrélations de la veille d'Azure.

L'en-tête dira « service arrêté » : la base rapatriée est une photo, le service tourne sur la machine. Son
état réel :

```bash
ssh strate-vm 'cd strate && ~/.local/bin/uv run strate veille status'
```

**Mettre à jour le code** (session SSH) :

```bash
cd ~/strate && git pull && uv sync --locked && systemctl --user restart strate-veille
```

Si la mise à jour change l'unité systemd elle-même, relancer plutôt `uv run strate veille install`.

**Surveiller le disque.** Le cache d'images optiques (`data/cache/optical`) n'est jamais purgé. Sur le Mac,
il pèse 5 Go pour 161 mines (31 Mo par mine en moyenne). Chaque mine nouvellement citée dans l'actualité
peut y ajouter son historique. Une fois par semaine :

```bash
df -h ~ && du -sh ~/strate/data/cache/optical
```

Au-delà de 80 % du disque, supprimer `data/cache/optical` (les signaux déjà calculés restent dans
`data/signals`) ou agrandir le disque dans le portail.

## 8. Coûts : ne jamais avoir de surprise

- **Alerte de budget** : portail → *Abonnements* → l'abonnement → *Budgets* → *Ajouter*. Montant mensuel
  de 5 €, alertes à 50 % et 100 % par courriel. Une alerte prévient mais n'arrête rien, et les chiffres
  arrivent avec 8 à 24 h de retard.
- **Arrêter la machine** : bouton *Arrêter* du portail. La machine passe en « arrêtée (libérée) » et le
  calcul n'est plus facturé. Un `sudo shutdown` lancé dans la machine, lui, laisse la facturation courir.
  Le disque reste facturé tant qu'il existe.
- **Tout supprimer** : supprimer le *groupe de ressources* créé avec la machine. Machine, disque, adresse
  IP et réseau partent ensemble.
- **L'adresse IP publique** n'est peut-être pas couverte par l'offre gratuite : vérifier dans *Analyse des
  coûts* après 48 h.
- **Fin des 12 mois gratuits** : la machine, le disque de 64 Gio et l'adresse IP publique deviennent
  payants. Vérifier le montant dans *Analyse des coûts* avant l'échéance. Avec Azure for Students,
  l'abonnement est suspendu à la fin du crédit, sans facture.

Les offres et leurs limites changent. Ce guide les a vérifiées sur les pages de Microsoft en novembre 2026.
Pour suivre ce qui est consommé : portail → *Abonnements* → *Analyse des coûts*.

## 9. Arrêter la veille

```bash
cd ~/strate && uv run strate veille uninstall   # arrête le service et retire l'unité
```

Les données restent dans `~/strate/data`.
