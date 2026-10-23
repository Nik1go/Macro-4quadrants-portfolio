import itertools
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import (ACOConfig, benchmark_target, calendar_years, optimize_aco,
                  prepare_data, read_mid60_csv, return_metrics, solution_rank, wealth_curves)


class OptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(2026)
        cls.returns=pd.DataFrame(rng.normal(np.linspace(.0001,.0008,6),np.linspace(.006,.025,6),size=(300,6)),columns=list('ABCDEF'))

    def test_against_exhaustive_small_problem(self):
        # Un oracle exhaustif indépendant sur les 20 paniers possibles.
        target=.06
        oracle=[]
        for selection in itertools.combinations(self.returns.columns,3):
            daily=self.returns[list(selection)].mean(axis=1)
            stats=return_metrics(daily,1.2)
            oracle.append((solution_rank(stats['cagr'],stats['volatility'],target),selection))
        expected=min(oracle)[1]
        config=ACOConfig(k=3,n_ants=30,n_iterations=30,seed=17)
        actual=optimize_aco(self.returns,target,1.2,config)
        self.assertEqual(tuple(actual['tickers']),expected)
        self.assertTrue(actual['feasible'])

    def test_infeasible_search_is_explicit(self):
        result=optimize_aco(self.returns,100.,1.2,ACOConfig(k=3,n_iterations=5,n_ants=10))
        self.assertFalse(result['feasible'])
        self.assertTrue((result['history']['shortfall']>0).all())

    def test_deterministic_and_best_feasible_monotone(self):
        config=ACOConfig(k=3,n_iterations=15,n_ants=20)
        a=optimize_aco(self.returns,.01,1.2,config)
        b=optimize_aco(self.returns,.01,1.2,config)
        self.assertEqual(a['tickers'],b['tickers'])
        pd.testing.assert_frame_equal(a['history'],b['history'])
        feasible=a['history'].loc[a['history'].feasible,'volatility']
        self.assertTrue((feasible.diff().dropna()<=1e-12).all())
        self.assertTrue(a['pheromones'].between(config.tau_min,config.tau_max).all())
        self.assertEqual(len(set(a['tickers'])),3)

    def test_invalid_data_and_parameters(self):
        for config in [ACOConfig(k=7), ACOConfig(k=3,rho=0), ACOConfig(k=3,n_elite=50), ACOConfig(k=3,beta=float('nan'))]:
            with self.assertRaises(ValueError):
                optimize_aco(self.returns,.01,1.2,config)
        invalid=self.returns.copy(); invalid.iloc[0,0]=np.nan
        with self.assertRaises(ValueError):
            optimize_aco(invalid,.01,1.2,ACOConfig(k=3))

    def test_feasible_always_precedes_lower_risk_infeasible(self):
        self.assertLess(solution_rank(.1,.9,.1),solution_rank(.099,.001,.1))


class FinancialDataTests(unittest.TestCase):
    def setUp(self):
        index=pd.bdate_range('2022-01-03',periods=300)
        self.universe=pd.DataFrame({'ticker':[f'S{i}' for i in range(11)],'name':[f'Stock {i}' for i in range(11)]})
        self.prices=pd.DataFrame(100*np.cumprod(np.full((300,13),1.0001),axis=0),index=index,columns=self.universe.ticker.tolist()+['^FCHI','^CM100'])

    def test_complete_calendar_and_explicit_exclusion(self):
        self.prices.loc[self.prices.index[5],'S0']=np.nan
        stocks,bench,audit=prepare_data(self.prices,self.universe)
        self.assertEqual(len(stocks.columns),10)
        self.assertFalse(audit.iloc[0].eligible)
        self.assertEqual(audit.iloc[0].missing_sessions,1)
        self.assertEqual(len(stocks),300)
        self.assertEqual(len(bench),300)

    def test_missing_index_never_silently_replaced_or_truncated(self):
        self.prices.loc[self.prices.index[0],'^CM100']=np.nan
        with self.assertRaisesRegex(ValueError,'incomplet'):
            prepare_data(self.prices,self.universe)

    def test_insufficient_universe_rejected(self):
        self.prices.loc[self.prices.index[0],['S0','S1']]=np.nan
        with self.assertRaisesRegex(ValueError,'au moins 10'):
            prepare_data(self.prices,self.universe)

    def test_drawdown_includes_initial_capital(self):
        stats=return_metrics(pd.Series([-.2,0.]),1.)
        self.assertAlmostEqual(stats['max_drawdown'],-.2)
        self.assertAlmostEqual(stats['cagr'],-.2)

    def test_exact_benchmark_winner_and_wealth(self):
        self.prices['^CM100']=np.linspace(100,150,300)
        stocks,bench,audit=prepare_data(self.prices,self.universe)
        target,winner,stats=benchmark_target(bench)
        self.assertEqual(winner,'CAC Mid 60')
        self.assertAlmostEqual(target,1.5**(1/calendar_years(bench.index))-1)
        tickers=stocks.columns[:10].tolist()
        curves=wealth_curves(stocks,bench,tickers)
        self.assertTrue((curves.iloc[0]==100).all())
        result=return_metrics(stocks[tickers].pct_change(fill_method=None).iloc[1:].mean(axis=1),calendar_years(stocks.index))
        self.assertAlmostEqual(curves.iloc[-1,0]/100-1,result['total_return'])

    def test_csv_simple_and_euronext(self):
        simple=read_mid60_csv('Date,Close\n2022-01-03,100\n2022-01-04,102\n')
        official=read_mid60_csv('Historical Data\nQS0010989117\nDate;Open;Close\n03/01/2022;99;100;\n04/01/2022;101;102;\n')
        pd.testing.assert_series_equal(simple,official)
        for text in ['Date,Close\n2022-01-03,-1\n','Date,Close\n2022-01-03,100\n2022-01-03,101\n']:
            with self.assertRaises(ValueError):
                read_mid60_csv(text)


if __name__=='__main__':
    unittest.main()
