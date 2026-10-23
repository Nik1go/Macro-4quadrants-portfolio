# Optimisation par colonie de fourmis

Sélection de 10 actions distinctes à poids constants de 10 % dans l'union du
CAC 40 et du CAC Mid 60. Le moteur minimise la volatilité historique annualisée
sous une contrainte de CAGR au moins égal au meilleur des deux indices prix.
Aucun solveur de poids n'est utilisé. Il s'agit d'une recherche approchée et
rétrospective, pas d'une garantie de rendement ni d'un optimum global certifié.

L'accueil et la barre latérale du portfolio donnent accès à la page, directement
accessible via `http://localhost:8502/?page=ant-colony-portfolio`.

## Données

Début demandé : 2022-01-01. La page publique lit uniquement l'instantané local
`snapshot/market_data.zip`, figé à la dernière clôture du **22/10/2026** lors de
sa création. La période réelle est affichée dans le laboratoire. La première
clôture commune sert de base 100. Le passage à une nouvelle date, un redémarrage
ou un effacement du cache ne déclenche aucun téléchargement.
Les actions et le CAC 40 (`^FCHI`) viennent de Yahoo (`Close`,
`auto_adjust=False`), sans réinvestissement des dividendes. Le CAC Mid 60
(`^CM100`) est complété, si nécessaire, par le graphique historique public
Euronext de l'indice prix exact (QS0010989117). Aucun ETF proxy.

Le parseur Euronext lit le format public utilisé par le JavaScript de la page.
Il est utilisé uniquement par l'outil local d'acquisition et le notebook.
Si ce format change ou si un indice reste incomplet, l'acquisition est bloquée.
Le parseur `Date,Close` du moteur permet une correction locale si nécessaire. Aucun cours n'est rempli
artificiellement. Les actions sans historique complet sont exclues explicitement.
Le snapshot `universe.csv` contient 100 membres officiels vérifiés le 23/10/2026,
leurs ISIN et les correspondances Yahoo. Il n'est pas actualisé automatiquement :
réviser ce snapshot pour une nouvelle composition. Les restructurations et
variations de plus de 50 % sont signalées dans l'audit et restent à contrôler.

L'instantané ZIP contient les clôtures, les métadonnées et l'univers utilisé.
Il doit être conservé avec les sources lors du déploiement. Il ne contient que
des CSV et du JSON. La page n'appelle jamais Yahoo ou Euronext, y compris si le
fichier est ancien, absent ou invalide : elle affiche alors une erreur locale.
Le bouton d'acquisition et l'import public ont été retirés ; le rechargement
global de la barre latérale est masqué sur cette page. Les autres projets du
portfolio conservent leur comportement propre.

Le cache Streamlit ne sert qu'à éviter de relire l'archive locale. Il est invalidé
si le fichier change. Les paramètres et la relance de la colonie restent
interactifs, avec des calculs sur les cours déjà enregistrés.

### Actualiser les cours en tant que propriétaire

Depuis un terminal du dépôt, uniquement lorsque la mise à jour est souhaitée :

```bash
airflow_venv/bin/python streamlit_app/ant_colony/refresh_snapshot.py
# Ou fixer une borne exclusive :
airflow_venv/bin/python streamlit_app/ant_colony/refresh_snapshot.py --end-exclusive 2026-10-23
```

Cette commande télécharge les cours, vérifie la période et les deux indices,
puis remplace l'archive atomiquement. Une acquisition invalide laisse le dernier
instantané exploitable intact. Il n'existe aucun endpoint public d'actualisation.
Le notebook reste un outil autonome : son code d'acquisition est exécuté sur
la machine de la personne qui le lance, pas par son simple téléchargement
depuis Streamlit.

## Hypothèses

- Équipondération maintenue par rééquilibrage quotidien théorique.
- Positions acheteuses, investissement intégral, aucun coût ni fiscalité.
- Volatilité : écart-type empirique quotidien multiplié par √252.
- CAGR : rendement composé annualisé par la durée calendaire / 365,25.
- Un admissible est toujours préféré à un non-admissible ; entre admissibles,
  on minimise le risque. Sans admissible trouvé, on minimise le déficit et on
  affiche explicitement que le seuil n'est pas atteint.
- Sélection sur toute la période et composition actuelle : biais rétrospectif
  et de survivance. Une mesure hors échantillon reste à construire.

## Exécuter

Depuis la racine du vrai dépôt WSL :

```bash
airflow_venv/bin/python -m pip install -r streamlit_app/ant_colony/requirements.txt
airflow_venv/bin/python -m streamlit run streamlit_app/app.py --server.port 8502
airflow_venv/bin/python -m unittest discover -s streamlit_app/ant_colony/tests -v
```

Le notebook `optimisation_portefeuille_fourmis.ipynb` contient tout le code et le
snapshot, sans import du dépôt. Ses sorties sont enregistrées sur données réelles.
Il peut être partagé seul et exécuté dans Jupyter ou VS Code avec les dépendances
indiquées dans sa première cellule. Il est aussi téléchargeable depuis la page.

Après modification du moteur ou du guide, régénérer puis exécuter le notebook :

```bash
airflow_venv/bin/python streamlit_app/ant_colony/build_notebook.py
```

Le guide, le pseudo-code et la notice suivent exactement `core.py`. Les traces
sont portées par les actions ; l'heuristique est l'inverse de la volatilité
individuelle et les dépôts sont classés par rang. C'est l'adaptation retenue pour
ce projet, avec une covariance de portefeuille implicite dans la série moyenne
des rendements.
