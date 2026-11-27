"""Veille : chaîne d'agents qui tourne en continu.

  1. collect     — récupère l'actualité minière mondiale (GDELT, toutes langues) et l'historique par mine
  2. normalize   — vérifie les formats, dédoublonne, rattache chaque article à une mine, le classe
  3. correlate   — met les mines en suivi, lance leurs analyses satellite, mesure les liens
                    entre événements et signaux satellite (avec intervalles de confiance)
  daemon         — enchaîne les trois étapes en boucle (service launchd sur le Mac)
Tout passe par la base partagée `store` (SQLite).
"""
