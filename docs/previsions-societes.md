# Prévisions et sociétés

Deux modules, mêmes règles que le reste du projet : sources ouvertes, aucune IA, chaque chiffre avec
son incertitude, et **un coefficient de corrélation affiché à chaque fois** qu'une relation est montrée.

## 1. Prévisions (`src/strate/forecast.py`)

Le passé sert à prédire le futur, et l'outil **s'évalue lui-même** en rejouant l'histoire.

- **Données** : séries annuelles de production par mine (`catalog.all_production()`, série principale
  choisie comme dans `screener.main_production`), emprise Maus, signaux satellite et actualité quand ils
  existent.
- **Modèles** (du plus simple au plus riche ; un modèle n'est retenu que s'il bat le naïf hors échantillon) :
  - `naif` : production de l'an dernier ;
  - `tendance` : tendance log-linéaire des 5 dernières années, amortie (φ choisi sur la période d'apprentissage) ;
  - `tendance+taille` : idem, rappelée vers l'ordre de grandeur donné par l'emprise (modèle d'emprise calibré).
- **Rétro-test « walk-forward »** : pour chaque mine et chaque année t de 2005 à 2021, on ne s'autorise que
  les données ≤ t − h pour prédire t (horizons h = 1, 2, 3 ans). Découpage temporel strict :
  choix des réglages ≤ 2014, évaluation 2015–2021 (jamais vue).
- **Mesures, par modèle et par horizon** (sur les variations log de production, et sur les niveaux) :
  `r` de Pearson et `rho` de Spearman prévision ↔ réel (IC 90 % par bootstrap sur les mines),
  `skill` = 1 − MSE(modèle) / MSE(naïf), erreur médiane en facteur, taux de bon sens de variation,
  couverture réelle des intervalles 80 % et 90 % (doit être proche de 80 % / 90 %), n mines, n points.
- **Intervalles de prévision** : quantiles empiriques des erreurs du rétro-test, par horizon (donc
  honnêtes par construction ; la couverture mesurée le vérifie).
- **Prévision d'avenir** : pour chaque mine à série suffisante, de la dernière année connue jusqu'à
  l'année en cours + 1, avec intervalles qui s'élargissent avec l'horizon. Écrit dans un **registre**
  `data/forecast/ledger.parquet` (`made_at, site_id, serie, target_year, model, value, lo80, hi80, lo90,
  hi90`) : chaque prévision est datée et ne se réécrit jamais, pour être notée quand le réel arrivera.
- `backtest() -> dict`, `forecast_site(site_id) -> dict`, `write_ledger() -> int`, CLI
  `strate forecast backtest|ledger`, résultats dans `data/forecast/backtest.json`.

## 2. Sociétés (`src/strate/companies.py`)

- **Entités** : sociétés propriétaires/exploitantes (liste suivie `sites.toml`, champs `owner`/`operator`
  du catalogue, Wikidata P127/P137), dédoublonnées par nom normalisé et QID ; identifiants : QID, CIK SEC,
  tickers. Lien société ↔ mines (avec part de détention quand connue).
- **Faits financiers et humains, par exercice** :
  - SEC EDGAR « companyfacts » (`https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`, ≤ 10
    requêtes/s, en-tête `User-Agent` obligatoire : valeur lue dans la variable `STRATE_SEC_UA`, sinon
    « Strate/0.1 open-source research tool » ; jamais d'adresse personnelle) — taxonomies `us-gaap` et
    `ifrs-full` (déposants 20-F/40-F) : CAPEX (`PaymentsToAcquirePropertyPlantAndEquipment`,
    `PurchaseOfPropertyPlantAndEquipment…`), chiffre d'affaires, dépenses d'exploration, actifs miniers,
    effectifs (`dei:EntityNumberOfEmployees` quand déclaré) ; correspondance nom ↔ CIK via
    `https://www.sec.gov/files/company_tickers.json` (et `company_tickers_exchange.json`) ;
  - Wikidata : effectifs (P1128) et chiffre d'affaires (P2139) datés (P585), pour les sociétés hors SEC.
  - Chaque fait garde sa source, son unité, sa devise, sa date de fin d'exercice et le formulaire.
- **Investissements annoncés (« dans quoi ils investissent »)** : à partir des titres collectés par la
  veille (`raw_docs`), extraction par règles multilingues d'un montant + devise + verbe d'investissement,
  classé en `expansion`, `exploration`, `usine/traitement`, `décarbonation/énergie`, `rachat/fusion`,
  `infrastructure`, `autre` ; rattaché à la société (nom) et/ou à la mine (rattachement de la veille).
- **Corrélations (toujours avec coefficient et IC)** : par société et sur l'ensemble des sociétés ayant
  assez d'années : CAPEX(t) ↔ production(t…t+2) des mines détenues (pondérée par la part), effectifs ↔
  production, CAPEX ↔ croissance de l'emprise satellite ; `r`, `rho`, IC 90 % bootstrap, n.
- `refresh(max_companies=…) -> dict` (borné, reprend où il s'est arrêté), `company(id) -> dict`,
  `table() -> DataFrame`, CLI `strate companies refresh`. Données dans `data/companies/`.

## 3. API (ajouts à `src/strate/api.py`)

| Route | Réponse |
|---|---|
| `GET /api/forecast/backtest` | `{"run_at", "periode":{"reglage":"≤2014","test":"2015–2021"}, "rows":[{model, horizon, n_mines, n_points, r, r_lo, r_hi, rho, rho_lo, rho_hi, skill, facteur_median, bon_sens, couverture80, couverture90}], "retenu":{horizon: model}, "methode": str}` |
| `GET /api/forecast/site?site=` | `{"serie", "historique":[{year, value}], "prevision":[{year, model, value, lo80, hi80, lo90, hi90}], "qualite":{r, rho, skill, facteur_median, couverture90, n} }` (null si pas de série) |
| `GET /api/companies?q=&pays=&limit=` | `{"total", "rows":[{id, nom, pays, tickers, cik, n_mines, capex_dernier, capex_annee, effectifs, effectifs_annee, ca_dernier, devise, investissements_12m}]}` |
| `GET /api/company?id=` | `{"profil":{…}, "mines":[{site_id, nom, part}], "faits":[{indicateur, annee, valeur, unite, devise, source, formulaire}], "investissements":[{date, montant, devise, categorie, titre, url, site_id}], "correlations":[{x, y, decalage, r, r_lo, r_hi, rho, n}]}` |
| `GET /api/companies/correlations` | `{"rows":[{x, y, decalage, r, r_lo, r_hi, rho, rho_lo, rho_hi, n_societes, n_points, verdict}], "methode"}` |

L'interface lit uniquement ces champs (plus les routes existantes).
