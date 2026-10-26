"""Générer le notebook autonome à partir du moteur et du guide, sans copie divergente."""
from pathlib import Path
import base64
import re
import nbformat as nbf

HERE=Path(__file__).parent


def build():
    import importlib.util
    spec=importlib.util.spec_from_file_location('aco_guide',HERE/'guide.py')
    guide=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guide)
    md=nbf.v4.new_markdown_cell
    code=nbf.v4.new_code_cell
    cells=[md("# Optimisation de portefeuille par colonie de fourmis\n\n"
              "**Léo Bertrand · Finance quantitative · Corrections du 26 octobre 2026**\n\n"
              "10 actions à 10 %, univers CAC 40 ∪ CAC Mid 60, depuis le 01/01/2022.\n\n"
              "Ce notebook est **autonome** : les cours corrigés, l'univers et le moteur sont inclus, aucun import du dépôt n'est nécessaire. "
              "Les sorties enregistrées proviennent du snapshot corrigé. *Restart Kernel & Run All* reproduit ces données sans réseau. "
              "Pour actualiser les cours, activer explicitement `REFRESH_DATA` ; Internet est alors nécessaire.\n\n"
              "Le code est généré à partir du même moteur que la page Streamlit. Le notebook ne contient aucun secret."),
           md("## 1 · Comprendre le problème\n"+guide.INTRO),
           md(r"On cherche $\min_{S\subset U, |S|=10}\sigma(S)$ sous $g(S)\geq \max(g_{CAC40},g_{Mid60})$. "
              r"Avec $r_{S,t}=\frac{1}{10}\sum_{i\in S}r_{i,t}$, "
              r"$g(S)=\exp(\sum_t\log(1+r_{S,t})/Y)-1$ et $\sigma(S)=s(r_S)\sqrt{252}$."),
           md("## 2 · Hypothèses et limites\n"+guide.CONVENTIONS),
           md("## 3 · Pseudo-code commenté\n```text\n"+guide.PSEUDOCODE+"\n```\n"+guide.METHOD_NOTES),
           md("## 4 · Notice des paramètres\n\n| Variable | Défaut | Signification | Effet |\n|---|---|---|---|\n"+
              "\n".join("| "+" | ".join(value.replace('|','/') for value in row)+" |" for row in guide.PARAMETERS)),
           md("## 5 · Environnement Python\n\nPython 3.10 ou supérieur. Décommenter la cellule suivante uniquement si les bibliothèques ne sont pas déjà installées."),
           code("# %pip install numpy pandas yfinance requests beautifulsoup4 cryptography matplotlib"),
    ]
    parts=re.split(r'^# %% (.+)$',(HERE/'core.py').read_text(encoding='utf-8'),flags=re.M)
    cells.append(code(parts[0].strip()))
    cells.append(md("## 6 · Univers d'investissement\n\nSnapshot officiel Euronext du **23/10/2026** : 40 + 60 titres. "
                    "Les ISIN permettent de contrôler les correspondances Yahoo. Les titres trop récents ou incomplets seront exclus explicitement. "
                    "Utiliser cette composition actuelle depuis 2022 introduit un biais de survivance."))
    csv=(HERE/'universe.csv').read_text(encoding='utf-8')
    cells.append(code('# Snapshot intégré pour que ce fichier reste partageable seul.\nUNIVERSE_CSV = """\n'+csv+'"""\nuniverse = pd.read_csv(StringIO(UNIVERSE_CSV))\nassert len(universe) == 100 and universe["ticker"].is_unique\ndisplay(universe.groupby("index").size().rename("Nombre de titres"))\ndisplay(universe.head())'))
    for title,body in zip(parts[1::2],parts[2::2]):
        cells.append(md("### "+title+"\n\n"+{
            'Données et contrôle qualité':"Yahoo fournit les clôtures des actions et du CAC 40. Le graphique **public** Euronext fournit l'indice exact si nécessaire, ainsi que les historiques cohérents de Vivendi et d'URW. Les corrections ponctuelles et les exclusions qualité sont explicites. Aucun cours manquant n'est rempli. Le parseur de CSV permet un import manuel de l'indice exact.",
            'Mesures financières':"Le CAGR utilise les années calendaires. La volatilité utilise 252 séances par an. La baisse maximale inclut le capital initial pour ne pas ignorer la première perte.",
            'Paramètres et classement des solutions':"Le classement est strict : un panier admissible passe toujours devant un panier sous le seuil. Entre admissibles, on minimise la volatilité. En absence d'admissible, on minimise d'abord le déficit de CAGR.",
            'Algorithme des fourmis':"Une phéromone par action. Le tirage pondéré sans remise correspond au pseudo-code séquentiel. Les paniers déjà vus sont mémorisés pour éviter de recalculer leurs métriques. Les dépôts sont classés par rang, puis bornés. Aucun solveur et aucune optimisation des poids."
        }.get(title,'')))
        cells.append(code(body.strip()))
    cells.append(md("### Ratios ajustés du risque\n\nTaux sans risque et cible du Sortino : 0 % par an. Même fonction que l'application."))
    cells.append(code((HERE/'ratios.py').read_text(encoding='utf-8')))
    embedded = base64.b64encode((HERE/'snapshot'/'market_data.zip').read_bytes()).decode('ascii')
    load_snapshot = (
        "# False : reproduire le snapshot embarqué sans réseau. True : acquisition explicite.\n"
        "REFRESH_DATA = False\n"
        "if REFRESH_DATA:\n"
        "    prices_all, metadata = fetch_market_data(universe)\n"
        "else:\n"
        "    from io import BytesIO\n"
        "    from zipfile import ZipFile\n"
        f"    EMBEDDED_SNAPSHOT = {embedded!r}\n"
        "    with ZipFile(BytesIO(base64.b64decode(EMBEDDED_SNAPSHOT))) as archive:\n"
        "        prices_all = pd.read_csv(BytesIO(archive.read('prices.csv')), index_col='Date', parse_dates=True)\n"
        "        metadata = json.loads(archive.read('metadata.json'))\n"
        "        universe = pd.read_csv(BytesIO(archive.read('universe.csv')))\n"
    )
    cells.extend([
        md("## 7 · Paramétrer l'expérience\n\nLes paramètres modifiables sont regroupés ici. Pour comparer plusieurs réglages, conserver les mêmes données et varier la graine ou un paramètre à la fois."),
        code("config = ACOConfig(\n    k=10,                # 10 actions, donc 10 % chacune\n    n_ants=40,           # paniers proposés par cycle\n    n_iterations=80,     # nombre de cycles\n    alpha=1.0,           # influence de la mémoire\n    beta=2.0,            # influence de l'inverse de la volatilité\n    rho=0.20,            # 20 % des phéromones disparaissent par cycle\n    n_elite=5,           # meilleurs paniers du cycle autorisés à déposer\n    q=1.0,               # intensité totale des dépôts par cycle\n    tau_min=0.05,        # maintient une probabilité d'exploration\n    tau_max=5.0,         # limite la domination des traces\n    seed=42,             # reproductibilité\n)\n# Facultatif : chemin d'un CSV Date,Close de l'indice prix CAC Mid 60 exact.\nMID60_CSV = None\nprint(asdict(config))"),
        md("## 8 · Charger et auditer les données\n\nPar défaut, les clôtures corrigées embarquées sont chargées sans réseau. "
           "`REFRESH_DATA = True` déclenche une acquisition avec une borne exclusive en heure de Paris. "
           "Le 01/01/2022 étant fermé, la base est la première clôture commune disponible. Le périmètre, les estimations et les exclusions sont affichés."),
        code(load_snapshot + "if MID60_CSV is not None:\n    imported = read_mid60_csv(Path(MID60_CSV).read_text(encoding='utf-8-sig'))\n    prices_all['^CM100'] = imported.reindex(prices_all.index)\n    metadata['mid60_source'] = str(MID60_CSV)\nprices_all, corrections = correct_known_prices(prices_all)\nif corrections:\n    metadata['data_corrections'] = corrections\nprices, benchmarks, audit = prepare_data(prices_all, universe)\nprint(json.dumps(metadata, ensure_ascii=False, indent=2))\nprint(f'Période : {prices.index[0].date()} → {prices.index[-1].date()}')\nprint(f'{len(prices)} clôtures ; {prices.shape[1]} / 100 actions éligibles ; aucun remplissage des cours manquants')\ndisplay(audit.loc[~audit.eligible, ['ticker','name','first_quote','missing_sessions','reason']])\ndisplay(audit.loc[audit.large_moves > 0, ['ticker','large_moves','large_moves_note']])\ntarget, winner, benchmark_stats = benchmark_target(benchmarks)\nprint(f'Benchmark gagnant : {winner} ; CAGR minimal = {target:.2%}')\ndisplay(benchmark_stats.style.format('{:.2%}'))"),
        md("## 9 · Exécuter la colonie\n\nToutes les métriques sont estimées sur la période qui sert à sélectionner les titres. Les résultats suivants sont rétrospectifs."),
        code("daily_returns = prices.pct_change(fill_method=None).iloc[1:]\nyears = calendar_years(prices.index)\nresult = optimize_aco(daily_returns, target, years, config)\nprint('Objectif historique atteint' if result['feasible'] else 'Aucune solution admissible trouvée')\nprint(f\"Paniers distincts évalués : {result['unique_evaluated']}\")\nif not result['feasible']:\n    print(f\"Déficit de CAGR : {100*(target-result['metrics']['cagr']):.2f} points\")\nselection = universe.set_index('ticker').loc[result['tickers'], ['name','index']].copy()\nselection['weight'] = 1 / config.k\nassert len(selection) == 10 and selection.index.is_unique\nassert np.isclose(selection['weight'].sum(), 1)\ndisplay(selection.style.format({'weight':'{:.0%}'}))\ncomparison = pd.concat([pd.DataFrame({'Portefeuille ACO':result['metrics']}).T, benchmark_stats])\ndisplay(comparison.style.format('{:.2%}'))"),
        code("portfolio_returns = prices[result['tickers']].pct_change(fill_method=None).iloc[1:].mean(axis=1)\nprint('Ratios annualisés ; taux sans risque et cible Sortino : 0 % par an')\ndisplay(pd.Series(portfolio_ratios(portfolio_returns), name='Portefeuille ACO'))"),
        md("## 10 · Comparer les trajectoires\n\nBase 100 sur la première clôture commune ; rééquilibrage quotidien, poids constants, frais et dividendes réinvestis exclus. Le CAC Mid 60 comporte une estimation au 28/11/2023, décrite dans les métadonnées."),
        code("import matplotlib.pyplot as plt\ncurves = wealth_curves(prices, benchmarks, result['tickers'])\nplt.style.use('seaborn-v0_8-whitegrid')\nfig, ax = plt.subplots(figsize=(12, 5))\ncurves.plot(ax=ax, color=['#087f8c','#bc7c19','#5464b2'], linewidth=1.8)\nax.axhline(100, color='grey', linestyle=':', linewidth=1)\nax.set(title='Optimisation rétrospective · base 100', xlabel='Date', ylabel='Valeur du portefeuille')\nax.legend(loc='upper left')\nfig.tight_layout()\nplt.show()"),
        md("## 11 · Lire la convergence\n\nLa volatilité du meilleur **admissible** ne peut qu'améliorer son classement au fil des cycles. Avant la première solution admissible, le déficit de CAGR prime. Le nombre de fourmis admissibles illustre le comportement de la colonie."),
        code("history = result['history']\nfig, axes = plt.subplots(1, 2, figsize=(12, 4))\naxes[0].plot(history['iteration'], 100*history['volatility'].where(history['feasible']), color='#087f8c')\naxes[0].set(title='Meilleur panier admissible', xlabel='Itération', ylabel='Volatilité annualisée (%)')\naxes[1].plot(history['iteration'], history['feasible_ants'], color='#bc7c19')\naxes[1].set(title='Fourmis respectant le rendement minimal', xlabel='Itération', ylabel='Nombre de fourmis')\nfig.tight_layout()\nplt.show()\ndisplay(result['pheromones'].head(20).rename('Phéromone finale'))"),
        md("## 12 · Exporter et prolonger l'étude\n\nMettre `EXPORTER = True` pour écrire les fichiers dans le dossier du notebook. "
           "Pour une étape suivante : comparer plusieurs graines à budget identique, introduire une validation chronologique hors échantillon, "
           "reconstruire les compositions historiques et intégrer les coûts de rééquilibrage. Un solveur de poids serait un autre problème d'optimisation."),
        code("EXPORTER = False\nif EXPORTER:\n    destination = Path('resultats_fourmis')\n    destination.mkdir(exist_ok=True)\n    selection.to_csv(destination / 'selection.csv')\n    curves.to_csv(destination / 'trajectoires.csv', index_label='Date')\n    audit.to_csv(destination / 'audit.csv', index=False)\n    history.to_csv(destination / 'convergence.csv', index=False)\n    pd.concat([prices,benchmarks],axis=1).to_csv(destination / 'clotures.csv',index_label='Date')\n    (destination / 'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2), encoding='utf-8')\n    (destination / 'parametres.json').write_text(json.dumps(asdict(config),indent=2), encoding='utf-8')\n    print(destination.resolve())"),
        md("## Références\n"+guide.SOURCES),
    ])
    nb=nbf.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python','version':'3.12'},
    })
    nbf.validate(nb)
    path=HERE/'optimisation_portefeuille_fourmis.ipynb'
    nbf.write(nb,path)
    print(f'Notebook autonome : {len(cells)} cellules → {path}')
    return path


if __name__ == '__main__':
    build()
