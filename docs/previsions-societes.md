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
- **Règles du rétro-test** : un seul φ et un seul κ (rappel λ_h = 1 − (1 − κ)^h vers l'emprise), réglés
  sur la moyenne des MSE relatives au naïf des horizons 1 à 3, cibles ≤ 2014 seulement. Un modèle n'est
  « retenu » pour un horizon que si la borne basse de l'IC 90 % de son skill est > 0 ; sinon le naïf reste
  retenu. `r` et `rho` portent sur les **variations** du log de la production depuis la dernière valeur
  connue : un modèle qui ne prévoit aucune variation (le naïf, ou `tendance` réglé sur φ = 0) n'en a pas
  (null, avec une `note`) ; `r_niveau` / `rho_niveau` mesurent les niveaux (log), dominés par les écarts de
  taille entre mines.
- **Limite connue (anticipation)** : l'emprise Maus est tracée sur des images 2019 ; pour les années
  anciennes elle vient « du futur ». Le rappel vers l'emprise de `tendance+taille` n'est donc pas
  strictement temporel, contrairement au reste du rétro-test (`backtest.json` › `avertissements`).
- `backtest() -> dict`, `forecast_site(site_id) -> dict` (**lecture seule** : sans `backtest.json`, rend
  l'historique et un `message`, ne lance jamais le rétro-test), `write_ledger() -> int`, CLI
  `strate forecast backtest|ledger` (le tableau affiché donne r, ρ, skill et r niveau avec leur IC 90 %,
  n mines et n points), résultats dans `data/forecast/backtest.json`.

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
  production, CAPEX ↔ croissance de l'emprise satellite ; `r`, `rho`, IC 90 % bootstrap, n. Elles portent
  sur les **variations annuelles en log** (croissance contre croissance), pas sur les niveaux, pour ne pas
  confondre deux tendances avec un lien (`companies.METHODE_CORRELATIONS`). Sur l'ensemble, l'IC
  rééchantillonne les **sociétés** : l'unité indépendante est la société (`n_societes`), pas le point.
  Un verdict par coefficient (`verdict_r`, `verdict_rho` ; `verdict` = `verdict_rho` pour l'ancien
  format) ; « démontré » exige un IC 90 % hors de 0, une q-valeur de Benjamini-Hochberg (`q_value`, sur la
  famille des tests du même calcul) < 0,10 et assez de sociétés.
- `refresh(max_companies=…) -> dict` (borné, reprend où il s'est arrêté), `company(id) -> dict`,
  `table() -> DataFrame`, CLI `strate companies refresh` (repli sur l'ancien `update(limit=…)` si le
  module n'a pas `refresh`). Données dans `data/companies/`.

## 3. API (ajouts à `src/strate/api.py`)

Lecture seule. Un module absent ou en panne donne une structure vide avec un `message`, jamais une erreur
serveur ; NaN et infinis deviennent `null`.

| Route | Réponse |
|---|---|
| `GET /api/forecast/backtest` | `{"run_at", "periode":{"reglage":"≤2014","test":"2015–2021"}, "rows":[{model, horizon, n_mines, n_points, r, r_lo, r_hi, rho, rho_lo, rho_hi, r_niveau, r_niveau_lo, r_niveau_hi, rho_niveau, rho_niveau_lo, rho_niveau_hi, skill, skill_lo, skill_hi, facteur_median, bon_sens, couverture80, couverture90, note}], "retenu":{horizon: model}, "methode": str}` |
| `GET /api/forecast/site?site=` | `{"site_id", "nom", "serie", "unite", "derniere_annee", "historique":[{year, value}], "prevision":[{year, horizon, mesure, model, value, lo80, hi80, lo90, hi90}], "qualite":Q, "qualite_par_horizon":[Q], "suivi_site":{model, horizon, n, facteur_median, portee:"mine"}, "backtest_run_at", "message"?}` ; null si pas de série |
| `GET /api/companies?q=&pays=&limit=` | `{"total", "rows":[{id, nom, pays, tickers, cik, n_mines, capex_dernier, capex_annee, effectifs, effectifs_annee, ca_dernier, devise, investissements_12m}]}` |
| `GET /api/company?id=` | `{"profil":{…}, "mines":[{site_id, nom, part}], "faits":[{indicateur, annee, valeur, unite, devise, source, formulaire}], "investissements":[{date, montant, devise, categorie, titre, url, site_id}], "correlations":[{x, y, decalage, n, r, r_lo, r_hi, rho, rho_lo, rho_hi, verdict_r, verdict_rho}]}` |
| `GET /api/companies/correlations` | `{"rows":[{x, y, decalage, r, r_lo, r_hi, rho, rho_lo, rho_hi, n_societes, n_points, q_value, verdict_r, verdict_rho, verdict}], "methode"}` |
| `GET /api/correlations/summary` | `{"rows":[{domaine, x, y, coefficient, type, lo, hi, n, unite_n, n_bootstrap_unit, unite_bootstrap, verdict, source}], "domaines":[{domaine, n, message}]}` |

Précisions :

- **`/api/forecast/site`** — `Q` = `{model, horizon, r, r_lo, r_hi, rho, rho_lo, rho_hi, r_niveau…,
  rho_niveau…, skill, skill_lo, skill_hi, couverture80, couverture90, facteur_median, n, n_mines, portee,
  note}`. `portee = "global"` : c'est le score du modèle retenu sur toutes les mines de la période de
  test, **pas** la précision propre à la mine (celle-ci est dans `suivi_site`, `portee = "mine"`).
  `horizon` = années après la dernière valeur connue ; `mesure` = vrai seulement si cet horizon a été
  mesuré hors échantillon dans le rétro-test (au-delà, l'intervalle est extrapolé). Sans `backtest.json`,
  la route rend l'historique, aucune prévision et un `message`. L'id demandé est essayé, puis ses
  jumelles (catalogue ↔ liste suivie), même si le premier essai échoue.
- **`/api/companies`** — `pays` : pays majoritaire des mines liées, sinon Wikidata P17, sinon null ; le
  filtre `pays=` compare les noms canoniques (alias, nom français, code ISO2, sans casse ni accent).
  `investissements_12m` : nombre d'annonces des 365 derniers jours (entier, 0 quand aucune ; null
  seulement si le module ne fournit pas la colonne). On compte, on ne somme pas : les devises diffèrent.
- **`/api/company`** — `categorie` vient de la catégorie de l'annonce (ou, ancien format, du type
  d'événement de la veille) ; `montant` et `devise` sont null quand le titre n'en donne pas.
  `correlations` : liste vide sans erreur quand la société a trop peu d'années.
- **`/api/correlations/summary`** — une ligne par coefficient. `type` : `r` (Pearson), `rho`
  (Spearman), `phi` (veille : mois d'événement × mois d'anomalie, 0/1) ou `lift` (risque relatif de
  Mantel-Haenszel). `n` et `unite_n` disent ce que n compte ; `n_bootstrap_unit` et `unite_bootstrap`
  disent combien d'unités indépendantes l'IC rééchantillonne :

  | domaine | n (`unite_n`) | unités du bootstrap | règle du verdict |
  |---|---|---|---|
  | calibration | mines, ou paires d'années pour la variation | mines (modèle d'emprise : IC analytique Fisher z) | IC 90 % seul (un test par coefficient publié) |
  | veille | mois-événements, ou mois avec actualité (volume) | mines | lift : verdict du module (q < 0,10, décalage 0, IC hors de 1) ; φ / ρ : `verdict_rho` du module |
  | sociétés | paires société-année | sociétés | `verdict_r` / `verdict_rho` du module (IC hors de 0, q < 0,10, assez de sociétés) |
  | prévisions | points mine-année (période de test) | mines | IC 90 % seul |

  Le verdict d'une ligne est toujours celui de **son** coefficient, jamais copié d'un autre coefficient
  de la même mesure. Quand le module n'a pas fourni ce verdict et que sa règle exige plus que l'IC
  (q-valeur, nombre d'unités), l'API en tire un de l'IC de la ligne qui ne dit jamais « démontré ».
  Prévisions : `x` = variation prévue, `y` = variation réelle du log de la production ; pour les modèles
  sans variation prévue, les lignes portent sur les niveaux (log). Des modèles aux mesures identiques
  (mêmes prévisions, par exemple `tendance` et `tendance+taille` quand le rappel vers l'emprise est nul)
  partagent une paire de lignes et y sont nommés ensemble.

L'interface lit uniquement ces champs (plus les routes existantes).
