"""Textes pédagogiques communs à la page et au notebook."""

INTRO = """
Une **métaheuristique** explore un très grand ensemble de solutions sans toutes les
énumérer. Ici, une solution est un panier de **10 actions distinctes**, choisi dans
l'union du CAC 40 et du CAC Mid 60. Les poids sont fixés : **10 % par action**.
L'algorithme optimise donc la **sélection des titres**, sans solveur de pondération.

L'objectif est de **minimiser le risque sous une contrainte de rendement
minimal**. Dans cette version, le proxy du risque est la vol annualisée des rendements
quotidiens. Le rendement minimal est le **CAGR** du CAC 40 ou du CAC Mid 60, selon le plus élevé. 

Les fourmis artificielles construisent des paniers au hasard, avec une préférence
pour les actions qui portent davantage de **phéromone** et dont la volatilité
individuelle est plus faible. On évalue ensuite le panier entier : les corrélations
comptent aussi. Les meilleurs paniers renforcent les traces des actions retenues,
tandis qu'une partie des anciennes traces s'évapore. Cette boucle alterne
exploration et apprentissage. Elle trouve une bonne solution approchée, **sans
certifier le minimum global**.
"""

CONVENTIONS = """
- **Période demandée :** du 01/01/2022 à la dernière clôture de l'instantané
  local affichée dans les résultats. Les cours de l'application sont figés :
  aucune acquisition n'est déclenchée par une visite ou une relance de la colonie.
  Le 1er janvier étant fermé, les performances commencent à la première clôture
  commune disponible. Une commande locale permet au propriétaire de préparer
  un nouvel instantané ; le notebook peut aussi télécharger des cours lorsqu'il
  est exécuté localement par son utilisateur.
- **Portefeuille :** 10 actions, positions acheteuses, poids constants à 10 %.
  La moyenne quotidienne des rendements suppose un **rééquilibrage quotidien
  théorique**, sans frais, fiscalité, contraintes de liquidité ni arrondis de titres.
- **Base de comparaison :** cours `Close` Yahoo (`auto_adjust=False`), sans
  réinvestissement des dividendes, comparés aux **indices prix** CAC 40 et CAC Mid 60.
  Les prix suivent les ajustements du fournisseur ; les opérations sur titres
  complexes doivent être contrôlées. `Adj Close` inclurait notamment des ajustements
  de dividendes et appellerait des indices de rendement total comparables.
- **Sources :** actions et CAC 40 via Yahoo Finance ; CAC Mid 60 via Yahoo, ou
  complément Euronext exact si l'historique Yahoo est incomplet. Aucun ETF proxy.
- **Qualité :** même calendrier que le CAC 40 ; une action sans historique complet
  est exclue avec un motif. Aucun remplissage artificiel des prix. Si l'un des
  indices manque, le calcul est arrêté jusqu'à une correction locale de l'instantané.
- **Univers :** composition officielle figée au 23/10/2026, conservée avec ISIN,
  source et symbole Yahoo. Ce n'est pas une reconstitution des compositions de 2022.
- **Interprétation :** sélection et mesure sur la même période, benchmark gagnant
  connu a posteriori et biais de survivance. C'est une **optimisation rétrospective**,
  pas un backtest prédictif. Une étape ultérieure devra sélectionner sur une fenêtre
  passée puis mesurer hors échantillon, avec compositions historiques et frais.
"""

PSEUDOCODE = """ENTRÉES : cours des N actions éligibles, CAC 40, CAC Mid 60, paramètres ACO
CONSTANTE : K = 10 ; chaque poids vaut 1 / K = 10 %

1. PRÉPARER LES DONNÉES
   Charger les clôtures depuis le 01/01/2022.
   # Application : instantané local, préparé manuellement hors interface publique.
   Garder les mêmes dates pour les actions et les deux indices exacts.
   Exclure et signaler toute action dont l'historique est incomplet.
   Calculer les rendements quotidiens et la durée Y en années calendaires.
   cible ← maximum(CAGR du CAC 40, CAGR du CAC Mid 60)

2. INITIALISER LA COLONIE
   rng ← générateur aléatoire fixé par seed                 # reproductibilité
   tau[i] ← 1 pour chaque action i                          # mémoire initiale
   eta[i] ← 1 / max(volatilité individuelle[i], 10^-8)       # information locale
   Normaliser eta par sa médiane ; meilleur ← aucun panier

3. POUR iteration ALLANT DE 1 À n_iterations
   Pour chaque fourmi parmi n_ants :
       panier ← ensemble vide
       Tant que le panier contient moins de K titres :
           Pour chaque action i non encore choisie :
               attraction[i] ← tau[i]^alpha × eta[i]^beta
           p[i] ← attraction[i] / somme des attractions restantes
           Tirer une action selon p, puis l'ajouter au panier # sans doublon

       r_panier[t] ← moyenne des K rendements au jour t       # poids fixes 10 %
       CAGR ← exp(somme(log(1 + r_panier)) / Y) - 1
       risque ← écart_type(r_panier, ddof=1) × racine(252)

       Si CAGR ≥ cible : classer par risque croissant        # contrainte stricte
       Sinon : classer par déficit (cible - CAGR) croissant, puis par risque
       Toujours préférer un panier admissible à un panier sous la cible.
       Mettre à jour le meilleur panier global si le classement s'améliore.

   tau ← (1 - rho) × tau                                   # évaporation
   élites ← les n_elite meilleurs paniers de cette itération
   Si le meilleur global est admissible : retirer les élites non admissibles
   dépôts ← [meilleur global] + élites
   Pour chaque panier de dépôts, de rang r allant de 0 à L - 1 :
       delta ← Q × (L - r) / (L × (L + 1) / 2)              # dépôt décroissant
       Pour chaque action i du panier : tau[i] ← tau[i] + delta
   tau ← borner(tau, tau_min, tau_max)                      # évite la stagnation
   Conserver CAGR, risque, déficit et nombre de solutions admissibles.

4. RESTITUER
   Si le meilleur respecte la cible : afficher « objectif historique atteint ».
   Sinon : afficher « aucune solution admissible trouvée », avec son déficit.
   Afficher les 10 titres, leurs poids, la convergence et les courbes base 100.
   Ne jamais présenter cette recherche approchée comme une preuve d'optimalité.
"""

PARAMETERS = [
    ("N", "Données", "Nombre d'actions ayant un historique complet ; au plus 100 dans le snapshot.", "Détermine le nombre de combinaisons possibles : C(N, 10)."),
    ("K / k", "10 (fixe)", "Nombre de titres distincts dans chaque panier.", "Poids w_i = 1/K, soit 10 % ; aucune vente à découvert."),
    ("r[i,t], r_panier[t]", "Calculés", "Rendement simple de l'action i et moyenne des 10 rendements au jour t.", "r[i,t] = P[i,t]/P[i,t-1] - 1 ; rééquilibrage quotidien implicite."),
    ("Y, 252", "Calculés / 252", "Y = jours calendaires / 365,25 ; 252 séances conventionnelles par an.", "Y annualise le CAGR ; 252 annualise la volatilité."),
    ("cible", "max(CAGR des indices)", "Rendement composé minimal à respecter sur la période observée.", "Contrainte calculée sur les mêmes clôtures et la même durée."),
    ("n_ants", "40", "Nombre de paniers construits à chaque itération.", "Davantage de fourmis élargit l'exploration mais augmente le temps de calcul."),
    ("n_iterations", "80", "Nombre de cycles construction, évaluation, évaporation, dépôt.", "Budget de 3 200 paniers tirés par défaut ; des doublons sont possibles."),
    ("tau[i] / τᵢ", "1 initialement", "Quantité de phéromone attachée à l'action i.", "Mémoire des paniers bien classés ; ce n'est ni un rendement ni un poids financier."),
    ("eta[i] / ηᵢ", "1 / volatilité", "Attractivité locale, normalisée par sa médiane.", "Favorise les titres moins volatils ; les covariances interviennent dans le risque du panier."),
    ("alpha / α", "1,0", "Influence de la phéromone dans la probabilité de tirage.", "α = 0 ignore la mémoire ; α élevé renforce l'exploitation."),
    ("beta / β", "2,0", "Influence de l'heuristique eta dans le tirage.", "β = 0 ignore eta ; β élevé privilégie les faibles volatilités individuelles."),
    ("rho / ρ", "0,20", "Fraction de phéromone qui disparaît à chaque cycle, dans ]0,1].", "ρ = 0,20 : 20 % s'évapore et 80 % reste, avant les nouveaux dépôts."),
    ("n_elite", "5", "Nombre maximal de paniers de l'itération autorisés à renforcer les traces.", "Le meilleur global dépose aussi ; les non-admissibles cessent de déposer dès qu'un admissible existe."),
    ("Q / q", "1,0", "Quantité totale déposée par cycle, répartie selon les rangs.", "Q règle le renforcement, en interaction avec rho et les bornes de tau."),
    ("L, r, delta / Δτ", "Calculés", "Nombre de dépôts, rang de dépôt (à partir de 0) et quantité déposée.", "Δτ = Q(L-r) / [L(L+1)/2] sur chaque action du panier."),
    ("tau_min, tau_max", "0,05 / 5,0", "Bornes positives appliquées aux traces après les dépôts.", "Empêchent une action de devenir totalement impossible ou excessivement dominante."),
    ("p[i]", "Calculée", "Probabilité conditionnelle de choisir une action restante.", "Proportionnelle à τᵢ^α ηᵢ^β ; recalculée après chaque choix sans remise."),
    ("seed", "42", "Graine du générateur aléatoire.", "Même graine + données + paramètres + environnement : même recherche."),
    ("10^-8, 10^-12", "Constantes numériques", "Plancher de volatilité de eta et tolérance de comparaison du CAGR.", "Évite la division par zéro ; tolérance de 10^-12 en rendement décimal."),
    ("ddof=1", "Convention statistique", "Écart-type empirique calculé avec un dénominateur T-1.", "Volatilité annuelle = écart-type quotidien × √252."),
]

SOURCES = """
- [Dorigo et Stützle · présentation de l'ACO](https://iridia.ulb.ac.be/comp2sys/research-aco.php.html)
- [Documentation yfinance.download](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html)
- [CAC 40 · composition Euronext](https://live.euronext.com/en/product/indices/FR0003500008-XPAR)
- [CAC Mid 60 · composition et historique Euronext](https://live.euronext.com/en/product/indices/QS0010989117-XPAR)
- [Règles de la famille CAC · indices prix, NR et GR](https://live.euronext.com/sites/default/files/documentation/index-rules2/CAC_Index_Family_Rulebook.pdf)

L'adaptation à une sélection de titres (trace par action, heuristique inverse de la
volatilité et dépôt par rang) est le choix de ce projet, pas une recette financière
imposée par la méthode ACO.
"""
