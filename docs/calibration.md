# Calibration des signaux satellite

*Recalculée le 2026-11-24 après deux relectures indépendantes (48 puis 18 défauts corrigés).
Tout se régénère avec `uv run strate calibrate report` : `data/calibration/report.json`,
`radar_report.json`, `radar_holdout.json`, `selection.csv` et la synthèse lue par l'interface,
`src/strate/data/calibration.json`.*

## En une phrase

**Depuis l'espace, Strate voit bien la taille d'une mine, et la taille dit l'ordre de grandeur de sa
production ; il ne voit pas, à ce jour, les hausses ou baisses de production d'une année sur l'autre.**

## Méthode

- **Vérité terrain** : production publiée (Jasansky et al. 2023, CC BY 4.0) — minerai extrait, métal
  récupéré ou charbon extrait selon la mine, rattachée à l'installation mère (seule porteuse de chiffres).
- **Fenêtre** : 2016–2019 (Sentinel-2 L2A dès 2016 ; la base de production s'arrête en 2019).
- **Échantillon** : 150 mines tirées au hasard ; recalées sur le catalogue corrigé, 135 restent éligibles
  (≥ 3 ans de production liée, emprise connue, pas uniquement souterraines — 436 éligibles aujourd'hui) ;
  15 exclues, chacune avec sa raison dans `selection.csv`. 129 ont assez de mois clairs (optique,
  326 paires d'années), 134 assez de passages (radar).
- **Satellite** : optique = la scène la plus claire de chaque mois, médiane annuelle des mois clairs ;
  radar = changement entre deux passages successifs de la même orbite (12 jours), médiane annuelle.
- **Intervalles** : à 90 %, bootstrap en rééchantillonnant les *mines* (les années d'une mine ne sont
  pas indépendantes).
- **Pas de réglage sur les données de test** : seuils et métriques choisis sur un groupe de mines,
  vérifiés sur l'autre ; pour le radar, choix figé avant de voir les mines de validation.

## Résultats

### 1. Taille ↔ échelle de production (en coupe, 107 mines)

| Ce qu'on compare à la production | Corrélation (IC 90 %) |
|---|---|
| Taille de la fenêtre d'analyse seule (tirée des contours Maus, **sans aucune image**) | 0,51 (0,37 – 0,63) |
| Surface nue Sentinel-2 dans cette fenêtre | 0,48 (0,35 – 0,60) |
| **Part de sol nu** (ce que l'image ajoute à la taille) | **0,17 (0,00 – 0,33)** |

Une version antérieure présentait la corrélation de la surface nue comme « validation du satellite » :
c'était faux, ce chiffre vient de la taille de l'emprise. L'image elle-même n'ajoute presque rien.

### 2. Variations d'une année sur l'autre

| Signal | Résultat | Verdict |
|---|---|---|
| Surface nue optique (toutes mines) | ρ = 0,11 (0,02 – 0,18) ; bon sens 54 % (48 – 60 %) | ≈ pile ou face |
| Seuils optiques réglés sur un groupe (ρ = 0,20), testés sur l'autre | ρ = 0,06 (−0,06 – 0,18) | **non** |
| Radar, métrique choisie sur un groupe, testée sur l'autre (73 mines) | ρ = 0,11 (−0,03 – 0,25) ; bon sens 56 % (47 – 64 %) ; 59 % (48 – 69 %) quand la production bouge de > 20 % | **non démontré** |

Le radar semblait faiblement validé (57 %) avant correction : 349 « paires » comparaient deux images
du même passage (changement ≈ 0) ; une fois retirées, aucun intervalle n'exclut le hasard.

### 3. Alertes

Une variation annuelle de plus de 10 % de la surface nue n'annonce une vraie variation de production
(> 20 %) que dans **29 % (23 – 35 %)** des cas, contre **29 %** au hasard. Les alertes mensuelles
affichées dans l'interface (scores z saisonniers persistants) **n'ont pas été évaluées** contre la
production : elles décrivent des changements physiques du site, rien de plus.

### 4. Radar pour la taille

Ajouter l'activité radar à l'emprise n'améliore pas l'estimation du tonnage (94 mines de validation,
jamais vues lors du choix de la métrique) : gain ×1,04 (0,93 – 1,08). **Rejeté.**

## Ce que Strate en fait

**Seule estimation publiée : l'ordre de grandeur de production d'après l'emprise réelle** (contours Maus
2019, aucune image Sentinel). Modèle log–log, erreur mesurée deux fois : validation croisée (une mine
retirée à la fois) et moitié des mines jamais vue par l'ajustement.

| Type | Mines | Corrélation | Erreur typique | 90 % des erreurs sous | Moitié tenue à l'écart |
|---|---|---|---|---|---|
| Minerai extrait | 245 | 0,58 | × 2,5 | × 8,5 | × 2,6 (90 % : × 8,4) |
| Charbon extrait | 165 | 0,41 | × 2,2 | × 8,8 | × 2,4 (90 % : × 9,2) |

Domaine : seulement les mines dont la commodité est connue et figure dans la population ajustée
(métaux ou charbon), jamais les usines, fonderies, raffineries, ni les emprises sans nom ni commodité
— d'où 3 904 sites estimés sur 79 562, plutôt que 25 000 estimations hors domaine. Population
d'ajustement : mines à production 2017–2020 connue (pas l'échantillon satellite ci-dessus).

## Corrections de données qui ont changé les chiffres

- **Charbon compté deux fois** (« Coal mined » + « Clean coal ») : tous les tonnages de charbon étaient ×2.
- **Production rattachée aux sous-installations** : 547 sites pointaient vers un identifiant sans
  production (Chuquicamata, Collahuasi, Los Bronces…) ; désormais rattachés à l'installation mère.
- **Métal additionné sur plusieurs bases** (minerai traité + concentré) : une seule base gardée, plus
  les cathodes par lixiviation (flux distinct).
- **Paires radar du même passage** : 349 paires corrompues retirées.
- **Composites optiques à 1 et 3 scènes mélangés** dans le cache : séparés (`s1c4/` pour la calibration).
- **Sélection figée avant corrections** : recalée sur le catalogue corrigé (15 mines exclues, dont
  Oyu Tolgoi, souterraine, et 12 qui lisaient la production d'une autre mine).
- **Mines productrices non rattachées** (Cerrejón, Ok Tedi… à 11–81 km de leur point ICMM) : rattachées
  par nom identique ; une usine sans production ne prend plus la place d'une mine productrice (Rio Tuba).

## Limites

- Production = teneur × débit × souterrain : invisible en surface.
- En plein désert, toute la fenêtre est « nue » : 10 fenêtres saturées exclues des modèles de surface.
- La base de production s'arrête en 2019 : rien n'est validé sur les années récentes.
- Une grande emprise (> 100 km²) peut regrouper plusieurs mines ou de l'orpaillage (avertissement ⚠).
- Pour 7 mines dont l'emprise a bougé de plus de 5 % entre les deux versions du catalogue, la fenêtre
  optique reste celle des images en cache (la recalculer demanderait de retélécharger les images).

## Pistes

- Vérité terrain récente : rapports trimestriels des sociétés cotées (2020 →).
- Mesurer l'activité autrement : cohérence interférométrique radar (SLC), chaleur (Landsat thermique),
  lumières de nuit, trafic de camions/trains/navires.
