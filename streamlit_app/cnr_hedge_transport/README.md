# Couverture des surcoûts de transport par le Brent

Le projet mesure le pied de facture gazole payé par le chargeur, puis étudie une
couverture avec des futures Brent. L'onglet **Étude Brent & hedge** utilise l'indice **CNR gazole professionnel mensuel officiel**, pour les
poids lourds d'au moins 7,5 tonnes, et le Brent BZ=F converti en EUR.

## Séries et méthode

- `data/cnr_gazole_professionnel_monthly.csv` contient les niveaux mensuels CNR
  de janvier 2019 à juillet 2026, transcrits du [tableau mensuel officiel CNR](https://www.cnr.fr/espaces/3/indicateurs/26/pdf?page=1), mis à jour en
  août 2026. La moyenne 2026 du tableau est exclue des observations mensuelles.
  Les valeurs bimensuelles de `CNR.csv` dans le dossier du
  notebook ne sont pas utilisées comme substitut aux valeurs mensuelles.
- `data/brent_bz_eur_monthly.csv` contient les moyennes mensuelles de
  `BZ=F / EURUSD=X` calculées à partir des clôtures quotidiennes communes Yahoo
  Finance, de janvier 2020 à juillet 2026. Le fichier a été produit le
  23 septembre 2026 par `refresh_brent_data.py`, avec au moins dix cotations
  quotidiennes par mois. Il s'agit d'un prix moyen en EUR par baril, pas du P&L
  d'une stratégie futures.
- Les régressions portent sur les variations mensuelles en pourcentage. Les
  décalages du Brent de 0 à 4 mois sont comparés sur **les mêmes 74 mois**,
  de juin 2020 à juillet 2026. Aucun mois manquant n'est interpolé.

Le meilleur ajustement dans cet échantillon est à **0 mois** :
**R² = 0,6858** et **β = 0,5809**. Le β est aussi le coefficient de variance
minimale pour les variations de l'indice et du Brent sur des notionnels
comparables. Il ne détermine pas directement un nombre de contrats, ni le
surcoût d'une facture : il faut appliquer la pondération CNR, le montant des
factures, la taille du contrat et les dates des positions.

Le notebook initial `hedgeTransport.ipynb` utilisait un proxy DGEC hors TVA et
décalait déjà le gazole d'un mois pour simuler la facture. Ses résultats
historiques (shift 1, R² = 0,5824, β = 0,3829) restent affichés séparément
pour comparaison ; ils ne servent plus aux métriques CNR principales.

La pondération gazole CNR est choisie selon le type de convoi et la date du
contrat. Le questionnaire simule douze mois de prestations, puis dimensionne
une position longue en mini Brent de 100 barils. Le P&L affiché reste un proxy
établi sur une série continue Brent en EUR ; il ne représente pas les mini-futures
réellement détenus et exclut les rolls, les frais et les besoins de marge.

```powershell
streamlit run streamlit_app/app.py
python -m unittest discover streamlit_app/cnr_hedge_transport/tests
python streamlit_app/cnr_hedge_transport/refresh_brent_data.py
```

Sources complémentaires : [CNR — Historique des pondérations indiciaires TRM 2026](https://www.cnr.fr/download/file/publications/CNR%20-%20Historique%20des%20pond%C3%A9rations%20indiciaires%20TRM%20-%202026.pdf),
[CME — future Brent BZ](https://www.cmegroup.com/markets/energy/crude-oil/brent-crude-oil-last-day.html).

## Exploration par quinzaine

La page affiche aussi les indices CNR gazole professionnel **bimensuels** des
jours 1-15 et 16-fin de mois, publiés de mars à juillet 2026 dans
`CNR_hedgetransport/CNR.csv`. Le fichier
`data/cnr_gazole_professionnel_fortnightly.csv` reprend ces 10 niveaux.
`data/brent_bz_eur_fortnightly.csv` contient, pour exactement les mêmes
quinzaines, les moyennes des clôtures quotidiennes communes BZ=F et EURUSD=X.
Pour actualiser ces moyennes après ajout de nouvelles lignes CNR, lancer
`python streamlit_app/cnr_hedge_transport/refresh_fortnightly_brent_data.py`.

Les shifts 0, 1 et 2 quinzaines sont comparés sur les **mêmes 7 variations**.
Le R² maximal descriptif vaut 0,7003 à shift 0, mais la courte fenêtre
ne permet pas de conclure que la fréquence bimensuelle améliore la précision
hors échantillon. Le coefficient utilisé pour illustrer le hedge reste donc
celui de la série mensuelle officielle. Le CNR précise que ses indices
mensuels demeurent les indices de référence.


## Questionnaire contractuel et mini Brent

La page **Contrat & couverture** distingue la **facturation** mensuelle ou
annuelle de la **révision du prix** mensuelle ou annuelle. La révision annuelle
peut utiliser le CNR du mois anniversaire ou la moyenne des douze mois ; les
factures mensuelles reçoivent alors une régularisation au dernier mois. Le
budget annuel est réparti également entre douze prestations. Le mois précédent
le début sert de base indiciaire. Le résultat montre la facture réelle du
scénario et son écart signé avec le budget initial.

Le coefficient Brent est estimé uniquement avec les données disponibles avant
le premier mois du contrat. La couverture vise tous les mois encore exposés ;
le nombre de mini-contrats est recalculé après chaque fixation mensuelle :

`mini cibles = budget restant × poids CNR × beta / (100 × prix Brent EUR)`

Le nombre de contrats est entier. Le questionnaire propose un arrondi au plus
proche, un arrondi inférieur sans surcouverture et un arrondi supérieur qui
maintient une position tous les mois mais peut surcouvrir fortement la fin
de la période. Un petit budget peut donc rendre impossible une couverture
exacte avec ce seul instrument. Le proxy de P&L utilise les changements de prix mensuels de la
série Brent continue en EUR. La série ne permet pas un backtest tradable.

Le [règlement CME MBC](https://www.cmegroup.com/rulebook/NYMEX/9/992.pdf)
confirme 100 barils par mini-contrat et un règlement fondé sur la moyenne
mensuelle du Brent ICE. MBC diffère donc de BZ=F utilisé dans la régression.
Le [bulletin CME](https://www.cmegroup.com/market-data/daily-bulletin.html)
consulté le 3 octobre 2026 indique zéro transaction MBC et 231 contrats
ouverts pour la séance du 2 octobre 2026 : la taille plus petite
ne garantit pas une bonne exécution pour une PME.

La [fiche ICE IMM](https://www.theice.com/publicdocs/circulars/15180_attach.pdf)
décrit aussi un mini Brent de 100 barils. La liquidité actuelle et les accès
courtiers de ce contrat restent à comparer avant toute recommandation.

Le [bulletin quotidien CME](https://www.cmegroup.com/market-data/daily-bulletin.html)
contient des règlements et volumes par échéance et reste disponible sans frais
après publication. Une collecte quotidienne à partir de ce bulletin peut
constituer un historique prospectif des prix MBC. Les archives historiques
complètes par échéance sont proposées par
[DataMine](https://www.cmegroup.com/market-data/real-time-and-historical-data.html),
qui indique un achat de données historiques ; leur disponibilité gratuite
sur toute la période de notre étude reste à confirmer. Aucun historique MBC
par échéance, aucun coût de roll vérifié et aucun tarif de courtage ne sont
encore présents dans ce dépôt.

Pour calculer le résultat réel, il faudra stocker les prix de chaque échéance
achetée et vendue, les dates et tailles des positions, le taux EUR/USD au
règlement, les commissions, le spread de transaction et les appels de marge.
Le différentiel entre deux échéances au moment du roll mesure la forme de la
courbe, mais ne constitue pas à lui seul un coût encaissé ou décaissé : le
P&L vient des variations des contrats effectivement détenus.

## Backtest du strip Brent ICE

L'onglet **Backtest ICE** utilise les séries quotidiennes `CO1` à `CO12` d'un
export Bloomberg personnel. Il suit pour chaque facture du mois M le future ICE
de livraison M+2 acheté en janvier : les changements de rang CO ne sont pas
des achats ou ventes. Le P&L de ce contrat est la somme des variations
quotidiennes `PX_LAST × barils`, converties en EUR au taux EUR/USD du jour ou au
taux constant choisi. La courbe compare les écarts de facture CNR cumulés, le
P&L de couverture et leur différence ; le tableau permet de vérifier chaque
échéance, quantité, date et prix. Une seconde courbe suit la position choisie
jour par jour.

Le nombre de barils est proposé à partir du β CNR / Brent en EUR **estimé
uniquement avant l'achat**, avec au moins 24 variations mensuelles et un mois
de recul supplémentaire faute de dates de publication CNR historiques :

`barils = facture initiale × poids gazole × β × EUR/USD à l'achat / prix ICE à l'achat`

La quantité peut être remplacée par une saisie manuelle. Il s'agit d'une
conversion indicative du β en risque financier, pas d'un optimum garanti pour
les P&L des échéances ICE. Le questionnaire **Contrat & couverture** conserve
son scénario mensuel sur mini Brent ; le backtest ICE est un calcul distinct.

L'application cherche `data/Brent Futures 2021.xlsx` dans son dossier local,
puis accepte un fichier importé par l'utilisateur. Ce classeur est exclu du
push public tant que les droits de redistribution de l'export Bloomberg ne sont
pas confirmés. Il pèse environ 623 Ko : Git LFS n'est pas utile pour sa taille.
Le déploiement public affiche donc un champ d'import. Le fichier est un proxy
de futures ICE fractionnaires : **0,1 ICE représente 100 barils de risque, mais
ne reproduit pas le règlement mensuel du MBC**. Les coûts, la marge et la date
effective de publication du CNR restent hors du modèle. Voir les spécifications
[ICE](https://www.ice.com/products/219) et [CME MBC](https://www.cmegroup.com/rulebook/NYMEX/9/992.pdf).
