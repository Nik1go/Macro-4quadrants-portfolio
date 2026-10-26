"""Vérifier hors réseau le snapshot, six graines et les échanges de un/deux titres.

Depuis la racine : airflow_venv/bin/python -m streamlit_app.ant_colony.audit_snapshot
"""
from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path
import numpy as np

from .core import ACOConfig, benchmark_target, calendar_years, optimize_aco, prepare_data, wealth_curves
from .ratios import portfolio_ratios
from .snapshot import read_snapshot


def local_swaps(returns, tickers, target, years):
    """Covariance pour filtrer par risque, puis CAGR exact pour chaque amélioration possible."""
    k = len(tickers)
    selected = np.array([returns.columns.get_loc(t) for t in tickers])
    outside = np.array([i for i in range(returns.shape[1]) if i not in selected])
    values = returns.to_numpy()
    cov = np.cov(values, rowvar=False, ddof=1)
    row_sums = cov[:, selected].sum(axis=1)
    base_variance = float(cov[np.ix_(selected, selected)].sum())
    base_daily_sum = values[:, selected].sum(axis=1)
    results = {}
    for size in (1, 2):
        additions = np.array(list(combinations(outside, size)))
        tested, improving, feasible_improving = 0, 0, 0
        for removed in combinations(selected, size):
            removed = np.array(removed)
            # Variance de la somme des K rendements après les échanges.
            constant = base_variance + cov[np.ix_(removed, removed)].sum() - 2 * row_sums[removed].sum()
            variances = np.full(len(additions), constant)
            for i in range(size):
                ai = additions[:, i]
                variances += cov[ai, ai] + 2 * row_sums[ai] - 2 * cov[np.ix_(ai, removed)].sum(axis=1)
                for j in range(i):
                    variances += 2 * cov[ai, additions[:, j]]
            tested += len(additions)
            candidates = additions[variances < base_variance - 1e-12 * k**2]
            improving += len(candidates)
            for added in candidates:
                daily = (base_daily_sum - values[:, removed].sum(axis=1) + values[:, added].sum(axis=1)) / k
                cagr = float(np.expm1(np.log1p(daily).sum() / years))
                feasible_improving += int(cagr + 1e-12 >= target)
        results[f'{size}_for_{size}'] = {
            'tested': tested, 'lower_risk': improving,
            'lower_risk_and_feasible': feasible_improving,
        }
    return results


def audit_snapshot():
    raw, metadata, universe = read_snapshot()
    prices, benchmarks, audit = prepare_data(raw, universe)
    daily = prices.pct_change(fill_method=None).iloc[1:]
    years = calendar_years(prices.index)
    target, winner, stats = benchmark_target(benchmarks)
    result = optimize_aco(daily, target, years, ACOConfig(seed=42))
    ratios = portfolio_ratios(daily[result['tickers']].mean(axis=1))
    others = {}
    for seed in (1, 2, 3, 7, 123, 2026):
        alternative = optimize_aco(daily, target, years, ACOConfig(seed=seed))
        others[str(seed)] = {'same_basket': set(alternative['tickers']) == set(result['tickers']),
                             'tickers': alternative['tickers'], 'metrics': alternative['metrics']}
    unconstrained = optimize_aco(daily, -1., years, ACOConfig(seed=42))
    history = result['history']
    pheromones = result['pheromones']
    validation = {
        'snapshot_last_session': metadata['snapshot_last_session'],
        'eligible_assets': len(prices.columns),
        'sessions': len(prices), 'tickers': result['tickers'],
        'metrics': result['metrics'], 'ratios_rf_0': ratios,
        'benchmark_stats': stats.to_dict(orient='index'), 'target': target, 'winner': winner,
        'unique_evaluated': result['unique_evaluated'],
        'first_final_best_iteration': int(history.loc[np.isclose(history.volatility, result['metrics']['volatility'], rtol=0, atol=1e-12), 'iteration'].iloc[0]),
        'new_baskets_last_iteration': int(history.unique_evaluated.diff().iloc[-1]),
        'pheromones_at_max': int(np.isclose(pheromones, 5.).sum()),
        'pheromones_at_min': int(np.isclose(pheromones, .05).sum()),
        'other_seeds': others,
        'without_return_floor': {'same_basket': set(unconstrained['tickers']) == set(result['tickers']),
                                 'tickers': unconstrained['tickers'], 'metrics': unconstrained['metrics']},
        'local_swaps': local_swaps(daily, result['tickers'], target, years),
    }
    destination = Path(__file__).parent / 'snapshot'
    (destination/'validation_2026-10-26.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    # Exports de référence pour la présentation, conservés dans le cache local.
    cache = Path(__file__).parent / 'cache'
    cache.mkdir(exist_ok=True)
    history.to_csv(cache/'initial_convergence.csv', index=False)
    audit.to_csv(cache/'audit.csv', index=False)
    wealth_curves(prices, benchmarks, result['tickers']).to_csv(cache/'trajectoires_fourmis.csv', index_label='Date')
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return validation


if __name__ == '__main__':
    audit_snapshot()
