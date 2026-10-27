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

D3 7.9.0 est embarqué dans `vendor/`, avec sa licence, pour supprimer la
dépendance à un CDN dans la page Streamlit. Aucune dépendance Python ajoutée ;
intégration vérifiée avec la version 1.53.1 de Streamlit installée dans WSL.
