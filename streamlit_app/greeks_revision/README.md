# Greeks Revision

Page du portfolio : `http://localhost:8502/?page=greeks`.

La carte « Greeks Revision » est disponible sur l’accueil et dans la sidebar.
`app_greeks.py` intègre `analyzer.html` dans un composant Streamlit v2 avec
styles isolés et hauteur automatique. Les colonnes et les graphiques s’adaptent
à la largeur disponible, y compris lors de l’ouverture de la sidebar.

Les calculs Black–Scholes et les interactions proviennent du fichier fourni
`greeks-analyzer (1).html`. Les changements portent sur le thème bleu nuit/cyan,
les libellés, l’accessibilité et l’intégration dans la navigation du portfolio.
Le Vega est exprimé par unité de volatilité et le Theta par année, conformément
aux calculs d’origine. La normalisation concerne uniquement les courbes
individuelles ; l’agrégat reste toujours brut. Le multiplicateur de contrat est 1.

Le graphique de payoff superpose les positions (pointillés colorés) et leur
somme (trait rouge épais), avec une légende sous le graphique. Les couleurs
correspondent au tableau et aux Greeks et restent stables lors d’une suppression.
Le payoff net à l’échéance inclut les primes : `(intrinsèque - prime) × sens × quantité`,
sans normalisation, ni capitalisation des primes, ni frais. Une prime unitaire
Black–Scholes est calculée séparément pour chaque option selon son type, son strike,
son échéance et sa volatilité, au spot S₀ et au taux r communs. S₀ et r sont des
paramètres de simulation : leurs changements recalculent toutes les primes,
y compris celles des positions existantes. Il ne s’agit pas de primes de marché
ou de prix d’exécution mémorisés. Le formulaire affiche la prime estimée avant
l’ajout ; le tableau et la légende affichent la prime par option. Le coût ou crédit
net des primes, quantités et directions comprises, figure sous le graphique.
Tous les strikes figurent dans l’échantillonnage pour conserver les cassures exactes.
L’axe des prix commence à zéro et couvre tous les strikes ainsi que S₀.
Si les échéances diffèrent, une note précise la convention de calcul du total.

D3 7.9.0 est embarqué dans `vendor/`, avec sa licence, pour supprimer la
dépendance à un CDN dans la page Streamlit. Aucune dépendance Python ajoutée ;
intégration vérifiée avec la version 1.53.1 de Streamlit installée dans WSL.
