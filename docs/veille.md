# Veille : la chaîne d'agents qui tourne en continu

```
   GDELT (toutes les 15 min, anglais + ~65 langues traduites)      API documentaire GDELT (historique par mine)
                │                                                              │
                ▼                                                              ▼
   ┌───────────────────────┐   raw_docs   ┌─────────────────────────┐   events   ┌──────────────────────────┐
   │ 1. COLLECTEUR          │ ───────────▶ │ 2. NORMALISEUR           │ ─────────▶ │ 3. CORRÉLATEUR             │
   │ collect.py             │              │ normalize.py             │            │ correlate.py               │
   │ récupère, garde        │              │ vérifie les formats,     │            │ met les mines en suivi,    │
   │ l'actualité minière    │              │ dédoublonne, rattache    │            │ lance leurs analyses       │
   │                        │              │ chaque article à une     │            │ satellite, mesure les      │
   │                        │              │ mine, le classe          │            │ liens événements ↔ signaux │
   └───────────────────────┘              └─────────────────────────┘            └──────────────────────────┘
                ▲                                                                              │
                └──────────── tracked (mines à approfondir : historique d'actualité) ◀─────────┘

   daemon.py enchaîne 1 → 2 → 3 toutes les 15 min (service launchd « com.strate.veille » sur le Mac).
   Tout passe par une base SQLite unique : src/strate/veille/store.py → data/veille/veille.db.
```

Règles communes : aucune IA (choix de l'utilisateur), sources ouvertes par API (GDELT, Wikidata),
**jamais** de pages derrière connexion ou paywall, aucune page d'article téléchargée (on ne garde que
les métadonnées GDELT : URL, titre, thèmes, lieux, organisations, langue, ton). Agent HTTP générique
« strate/0.1 (open-source research tool) », `Retry-After` respecté, réessais avec attente croissante.

## Contrat des étapes (Python)

Chaque module expose `run_once(con: sqlite3.Connection, **limites) -> dict` (compteurs pour
`pipeline_status`), idempotent, qui traite un lot borné puis rend la main. Il appelle
`store.set_status(con, '<étape>', ok, message, **compteurs)`. Une erreur réseau n'arrête jamais la
chaîne : elle est notée et retentée au cycle suivant.

### 1. `strate.veille.collect`

- `run_once(con, backfill_hours=48, max_files=40, doc_queries=6) -> dict`
- Flux GKG 2.0 : `http://data.gdeltproject.org/gdeltv2/lastupdate.txt` et `lastupdate-translation.txt`
  (les URL sont en http, le serveur redirige vers https). Fichiers `AAAAMMJJHHMMSS.gkg.csv.zip` et
  `AAAAMMJJHHMMSS.translation.gkg.csv.zip`, un par quart d'heure. **Les fichiers récents répondent
  souvent 404 pendant des heures** : statut `missing`, réessayé, abandonné après 72 h.
- Au premier lancement, rattrape `backfill_hours` d'historique ; ensuite tous les quarts d'heure
  manquants depuis le dernier fichier `done`.
- GKG 2.1, 27 colonnes séparées par des tabulations : 0 GKGRECORDID, 1 DATE, 3 SourceCommonName,
  4 DocumentIdentifier (URL), 8 V2Themes (`THEME,offset;…`), 10 V2Locations
  (`type#nom#pays#adm1#adm2#lat#lon#featureid#offset;…`), 11 V2Persons, 13 V2Organizations,
  15 V2Tone (1re valeur), 25 TranslationInfo (`srclc:xx;eng:…`), 26 Extras (`<PAGE_TITLE>…</PAGE_TITLE>`,
  `<PAGE_PRECISEPUBTIMESTAMP>…`).
- Garde un document s'il porte un thème minier (`ENV_MINING`, `WB_1699_METAL_ORE_MINING`,
  `WB_1698_COAL_MINING`, `WB_895_MINING_SYSTEMS`, `WB_1700_NONMETALLIC_MINERAL_MINING_AND_QUARRYING`,
  `TAX_FNCACT_MINER(S)`, `ENV_COAL`, `ENV_METALS`, `SLFID_MINERAL_RESOURCES`…) **ou** si son titre contient
  un mot « mine » multilingue (mine, mina, minería, mineração, Bergwerk, Grube, рудник, шахта, 矿, 鉱山,
  광산, tambang, maden, kopalnia…).
- Historique par mine (API documentaire `https://api.gdeltproject.org/api/v2/doc/doc`, **une requête
  toutes les 6 s au plus**, 429 → attendre) pour les mines de `tracked` sans `backfill_status` :
  `mode=timelinevolraw` (volume mensuel depuis 2017 → `news_volume`) et `mode=artlist`
  (`maxrecords=250`, `sort=datedesc` → `raw_docs` avec `hint_site`). Requête = nom(s) de la mine entre
  guillemets + « mine ». Noms refusés : sans mot distinctif, pays ou région seuls, et pays/région +
  relief, étendue ou substance anglaise (« Ural Mountains », « Lake Victoria », « Copper Chile ») sauf si
  le nom porte un mot d'ouvrage (« Turkey Creek Mine »). Un nom principal fait de mots de lieu (« Red
  Lake », « Cobre Panama ») est gardé en phrase exacte.
- `fetch_names(con) -> int` : noms de chaque mine du catalogue **dans toutes les langues** (libellés et
  alias Wikidata, par lots `VALUES`, en POST) → `data/veille/names.parquet` (`site_id, name, lang`),
  rafraîchi au plus une fois par semaine.

### 2. `strate.veille.normalize`

- `run_once(con, batch=5000) -> dict` : lit `raw_docs.status='new'`.
- **Vérification des formats** (sinon `rejected` + `reject_reason`) : URL http(s) valide, date
  analysable, ni future ni avant 2015, lat/lon dans les bornes, code langue ISO, titre non vide.
- **Dédoublonnage** : même URL (même `id`) ; même titre normalisé + même domaine à moins de 3 jours.
- **Rattachement à une mine** : index des noms (catalogue `name` + `aliases`, liste suivie, noms
  multilingues de `names.parquet`), comparaison sur les mots distinctifs (`catalog._tokens`, numéros de
  mine qui séparent). Un nom de mine doit apparaître **en entier** dans le titre ou les organisations.
  Les noms ambigus (plusieurs mines) sont tranchés par la géographie GDELT (lieu à < 50 km, sinon même
  pays), puis, parmi les homonymes restants (ceux des pays de l'article s'il en cite), par la seule fiche
  qui publie sa production, à défaut la seule qui a une emprise mesurée (méthode `…+priorité`, force
  0,7 pour un nom de plusieurs mots, 0,4 pour un mot seul, donc sous le seuil 0,5 sans lieu proche).
  Choix de jugement : une fiche sans données est le plus souvent mineure ou un doublon (« Escondida » :
  la mine de BHP est la seule de ses 4 homonymes à publier sa production). Deux fiches de même rang :
  pas de rattachement. Sur le catalogue de novembre 2026, 551 des 1 887 clés à homonymes sont ainsi
  tranchées sans lieu. `hint_site` (requête par mine) : rattachement
  direct, confiance plus basse si le nom n'apparaît pas. Le nom est cherché parmi TOUS les noms de la
  fiche visée, même ceux que l'index de recherche écarte (fiche non étayée, trop d'homonymes, double
  d'une autre fiche à moins de 10 km).
- **Noms faits de mots de lieu** (`normalize.place_key`) : les noms principaux que le collecteur
  cherche en phrase exacte (« Red Lake », « Copper Mountain ») et les noms région + relief (« Ural
  Mountains ») sont indexés, mais comme des mots courants : rattachés seulement avec un lieu GDELT
  ponctuel à moins de 50 km ou par la requête de la mine (« Red Lake » est aussi une ville).
- Après chaque lot, `meta` « doc_reconnu:<mine> » dit, pour chaque mine dont l'historique est fait, si
  l'un des noms cherchés par l'API est reconnu dans un titre (sur le catalogue de novembre 2026 : 412
  mines sur 56 014 ne le sont pas, contre 25 728 avant la correction).
- **Classement** (`store.EVENT_TYPES`) par thèmes GDELT (KILL, MANMADE_DISASTER*, UNSAFE_WORK_ENVIRONMENT
  → accident ; STRIKE, PROTEST, ECON_UNIONS → social ; LEGISLATION, ARREST, WB_849… → juridique ;
  ENV_* pollution → environnement ; ECON_BANKRUPTCY, ECON_STOCKMARKET, ECON_IPO… → finance ;
  TERROR, KIDNAP, ARMEDCONFLICT → securite) **et** mots-clés multilingues dans le titre (rachat,
  acquisition, adquisición, aquisição, Übernahme, поглощение, 收购 ; production, output, expansion,
  shutdown, suspend… ; tailings, relaves, rejeitos…). Un type par événement (le plus fort).
  Les verbes « mourir » qui meurent aussi au figuré (die, dies, meurt, muere, murió, morre, stirbt,
  muore…) ne comptent qu'avec une personne ou un nombre à deux mots au plus (« Two workers die »,
  « 3 die ») : « strike dies down », « deal dies », « Muere el acuerdo minero » ne sont pas des
  accidents. Arabe « توفي » en mot entier (pas « توفير », fournir). Choix : un titre sans sujet
  humain (« Muere en la mina ») est perdu, comme « öldü » turc ou « mati » indonésien.
- `confidence` = force du rattachement × force du classement (0–1), détail dans `evidence` (JSON).

### 3. `strate.veille.correlate`

- `run_once(con, new_per_day=20, analyses_per_run=2, recompute_hours=6) -> dict`
- **Suivi automatique** : mines avec des événements (confiance ≥ 0,5) dans les 30 derniers jours,
  classées par nombre × confiance ; au plus `new_per_day` nouvelles mines par jour dans `tracked`.
- **Analyses satellite** : pour au plus `analyses_per_run` mines suivies sans analyse récente, lance
  dans le processus `REGISTRY['optical'|'attention'|'radar'].update(...)` depuis 2020 (comme
  `api._analyse`), note `analysis_status`. Les mines analysées apparaissent alors d'elles-mêmes dans le
  radar et le screener.
- **Étude d'événements**, au plus toutes les `recompute_hours` : panel (mine × mois) des mines
  suivies/analysées ; anomalie satellite = alerte persistante retenue par `fusion.dossier`
  (|a_signal| ≥ 2) par signal ; événement = au moins un événement du type ce mois (confiance ≥ 0,5).
  **Couverture** : pour chaque mine, seuls comptent les mois où l'actualité était réellement collectée
  (depuis le premier mois du flux GKG traité ; pour une mine dont l'historique est « fait » et dont un
  nom cherché est reconnu par le normaliseur, depuis l'article le plus ancien renvoyé) — sinon des
  années « sans événement » faute de collecte, ou faute de nom reconnu, fausseraient la comparaison.
  Test : **statistique de Cochran–Mantel–Haenszel stratifiée par mine** (un tableau 2×2 par mine : mois
  d'événement ou non × anomalie ou non), **p-valeur par rotation circulaire** de la frise des événements
  dans chaque mine (loi exacte par convolution des mines). Les mois d'une mine ne sont pas
  indépendants : une anomalie retenue dure au moins deux mois, les événements arrivent en grappes. Le
  CMH exact, qui mélange les mois, rejetait trop souvent sous persistance (simulation, frises de Markov
  de 5 à 20 mines : 15 à 38 % de p < 0,10, et jusqu'à 89 % de calculs avec au moins un faux « lien
  démontré » sur 24 tests) ; la rotation garde l'autocorrélation des deux frises : 6 à 8 % de faux
  « lien démontré » pour un test, 7 à 10 % des calculs de 24 tests avec au moins un. Limite : elle
  suppose les événements stationnaires dans la fenêtre couverte. La p du CMH exact reste dans
  `details.p_cmh_mois_independants`. Effet = rapport de risques de Mantel–Haenszel (`lift`), IC 90 %
  par bootstrap sur les mines, correction de continuité de 0,5 sur chaque case quand aucune anomalie
  ne tombe hors des mois d'événement (une association parfaite donne un lift fini et élevé, pas une
  absence de résultat ; aucune coïncidence donne un lift exact de 0, non corrigé).
  **Famille principale = décalage 0 seulement** (type × signal, plus le volume d'actualité) : q-valeurs
  de Benjamini-Hochberg sur cette famille ; verdict `lien démontré` si q < 0,10 et IC hors de 1.
  Décalages −3 … +3 : calculés et stockés avec leur propre correction, verdict `exploratoire` (on ne
  déclare rien sur eux). **Mines informatives** : seules celles qui ont au moins un mois d'événement
  et au moins une anomalie (aucun des deux tous les mois) entrent dans le test, le lift, le phi et le
  bootstrap ; `n_mines` les compte, `n_events` compte leurs mois-événements (`details` garde
  `mines_avec_evenements` et `mois_evenements_total`). `trop peu de données` si moins de 5 mines
  informatives ou 10 mois-événements de ces mines : une seule mine informative parmi cinq mines à
  événements ne peut plus produire un verdict sur un IC réduit à un point.
  **Coefficient de chaque type** (colonnes `rho`, `rho_lo`, `rho_hi`) : phi intra-mine entre mois
  d'événement (0/1) et mois d'anomalie (0/1), sommes centrées dans chaque mine, IC 90 % par les mêmes
  tirages de mines que le lift ; son numérateur est la statistique du test (même p, même q).
  `verdict_rho` : même règle appliquée à l'IC du phi (hors de 0), jamais recopiée du verdict du lift.
  Ces taux valent pour le lift comme pour le phi (même p).
  Volume d'actualité (`news_volume`) ↔ signal : corrélation de rang intra-mine, p-valeur par loi
  normale de la rotation de la frise du volume dans chaque mine (le mélange des mois donnait 19 à 25 %
  de p < 0,10 sous H0 avec des frises AR(1) de coefficient 0,7 à 0,8 ; la rotation 9 à 11 % ; la p du
  mélange reste dans `details.p_permutation_mois_independants`), mêmes règles (`verdict_rho` =
  `verdict`). `details` dit le coefficient (`coefficient_rho`), ce que compte `n_events`
  (`unite_n`) et l'unité du bootstrap (`unite_bootstrap` = mines, `n_unites_bootstrap`).
  Écrit `correlations` (scope `global`). Le texte de méthode affiché vient de `correlate.METHODE`.

### `strate.veille.daemon`

- `run_forever(interval_s=900)` : boucle collect → normalize → correlate, isole les erreurs par étape,
  s'aligne sur le quart d'heure, s'arrête proprement sur SIGTERM, priorité basse (`os.nice`).
  Le statut écrit par une étape fait foi s'il est encore en base à la fin de l'étape
  (`store.watch_status`, variable de contexte propre à chaque fil) ; une étape qui l'a annulé par son
  propre rollback, ou qui n'en a pas écrit, est notée « terminé » par le service.
- CLI : `strate veille once | run | status | install | uninstall`. `install` écrit
  `~/Library/LaunchAgents/com.strate.veille.plist` (KeepAlive, journaux dans `logs/veille.log`) et le
  charge avec `launchctl` ; `uninstall` l'arrête et le retire.

## Contrat de l'API (FastAPI, `src/strate/api.py`)

| Route | Réponse |
|---|---|
| `GET /api/veille/status` | `{"stages":[{stage,last_run,ok,message,counters}], "totaux":{docs, docs_en_attente, rejetes, events, mines_avec_evenements, suivies, fichiers_faits, fichiers_manquants}, "service":{"actif":bool, "dernier_cycle":iso\|null}}` |
| `GET /api/veille/types` | `store.EVENT_TYPES` |
| `GET /api/veille/events?type=&site=&lang=&country=&since=&min_conf=0.5&limit=100` | `{"total":n, "events":[{id, site_id, site_name, event_type, published_at, title, url, domain, lang, country, lat, lon, confidence, match_method}]}` (plus récents d'abord) |
| `GET /api/veille/stats?days=30` | `{"par_type":{type:n}, "par_jour":[{jour, n}], "par_langue":{lang:n}, "par_pays":{pays:n}, "top_mines":[{site_id, site_name, n}]}` |
| `GET /api/veille/correlations` | `{"run_at":iso\|null, "methode":str, "rows":[{event_type, signal, lag_months, n_mines, n_events, lift, lift_lo, lift_hi, rho, rho_lo, rho_hi, p_value, q_value, verdict, verdict_rho, details}]}` (rho = phi intra-mine pour un type, rang intra-mine pour le volume) |
| `GET /api/veille/site?site=` | `{"events":[…200 derniers], "volume":[{month, n_articles}], "tracked":{…}\|null}` |
| `GET /api/veille/tracked` | `[{site_id, site_name, reason, since, last_event_at, events_count, analysis_status}]` |

L'onglet **Veille** de l'interface lit uniquement ces routes.
