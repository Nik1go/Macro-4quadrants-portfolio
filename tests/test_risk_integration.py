"""Check the original dark shell and the integrated Risk app with real data."""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / 'streamlit_app/app.py'
RISK = ROOT / 'streamlit_app/portfolio_risk'
TABS = ['Overview', 'Data', 'Performance', 'Correlation', 'VaR & ES', 'Backtesting', 'Stress tests', 'Concentration', 'Findings & export']


def widget(elements, label):
    return next(element for element in elements if element.label == label)


def prepare_rerun(at):
    # AppTest 1.53 expects a list for single-selection segmented controls.
    for control in at.get('button_group'):
        if isinstance(control.value, str):
            control.set_value([control.value])
    return at


def content_digest(source):
    tree = ast.parse(source)
    class VisualNormalizer(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            return None if node.name == 'style_fig' else self.generic_visit(node)
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                if '<style>' in node.value:
                    return ast.Constant(value='<STYLE>')
                if re.fullmatch(r'#[0-9a-fA-F]{3,8}', node.value) or node.value.startswith('rgba('):
                    return ast.Constant(value='<COLOR>')
            return node
    return hashlib.sha256(ast.dump(VisualNormalizer().visit(tree), include_attributes=False).encode()).hexdigest()


class RiskIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(RISK))
        import risk_engine
        cls.network = patch.object(risk_engine, '_yf_download', side_effect=AssertionError('Unexpected network access'))
        cls.network.start()

    @classmethod
    def tearDownClass(cls):
        cls.network.stop()
        sys.path.remove(str(RISK))

    def app(self, route=None):
        at = AppTest.from_file(str(APP), default_timeout=240)
        if route:
            at.query_params['page'] = route
        return prepare_rerun(at.run())

    def assert_healthy(self, at):
        self.assertEqual([], [element.message for element in at.exception])
        self.assertEqual([], [element.value for element in at.error])
        prepare_rerun(at)

    def test_01_risk_content_engine_notebook_and_data_are_unchanged(self):
        manifest = json.loads((RISK / 'source_sha256.json').read_text())
        for name, digest in manifest.items():
            if name != 'app.py':
                with self.subTest(file=name):
                    self.assertEqual(digest, hashlib.sha256((RISK / name).read_bytes()).hexdigest())
        content = json.loads((RISK / 'content_sha256.json').read_text())
        self.assertEqual(content['content_and_calculations'], content_digest((RISK / 'app.py').read_text()))

    def test_02_original_cards_and_home_restored_risk_last(self):
        at = self.app()
        self.assert_healthy(at)
        self.assertEqual('Finance Portfolio', at.title[0].value)
        self.assertTrue(any('Quantitative Analyst | Data Engineer' in element.value for element in at.markdown))
        cards = [element for element in at.button if element.key and element.key.startswith('btn_')]
        self.assertEqual(['btn_macro','btn_crypto','btn_pairs','btn_cnr','btn_dca','btn_mc','btn_opti','btn_poly','btn_aco','btn_greeks','btn_risk'], [element.key for element in cards])
        html = '\n'.join(element.value for element in at.markdown)
        self.assertIn('background-color: #0a0e27 !important', html)
        self.assertIn('portfolio-brand', html)
        self.assertNotIn('Financial intuition.', html)
        at.button(key='btn_risk').click().run()
        self.assert_healthy(at)
        self.assertEqual('risk', at.query_params['page'][0])
        self.assertEqual(TABS, [tab.label for tab in at.tabs])
        at.button(key='risk_home').click().run()
        self.assert_healthy(at)
        self.assertEqual('home', at.query_params['page'][0])

    def test_03_sidebar_component_selection_updates_route(self):
        # AppTest does not implement custom-component input. Simulate just the
        # component selection; the actual menu is also checked in the browser.
        def select_risk(**kwargs):
            self.assertEqual('Portfolio Risk Engine', kwargs['options'][-1])
            self.assertEqual(len(kwargs['options']), len(kwargs['icons']))
            if st.session_state.current_page != 'Portfolio Risk Engine':
                st.session_state[kwargs['key']] = 'Portfolio Risk Engine'
                kwargs['on_change'](kwargs['key'])
                st.rerun()
            return 'Portfolio Risk Engine'
        with patch('streamlit_option_menu.option_menu', side_effect=select_risk):
            at = self.app('home')
            self.assert_healthy(at)
            self.assertEqual('risk', at.query_params['page'][0])
            self.assertEqual('Portfolio Risk Engine', at.title[0].value)

    def test_04_direct_link_matches_original_results_and_dark_charts(self):
        integrated = self.app('risk')
        original_source = Path('/mnt/c/Users/leoja/Documents/Github/streamlit_app/app.py')
        reference = original_source if original_source.exists() else RISK / 'app.py'
        standalone = AppTest.from_file(str(reference), default_timeout=240).run()
        self.assert_healthy(integrated)
        self.assert_healthy(standalone)
        self.assertEqual(TABS, [tab.label for tab in integrated.tabs])
        self.assertEqual(16, len(integrated.get('plotly_chart')))
        self.assertEqual(20, len(integrated.dataframe))
        self.assertEqual(6, len(integrated.get('download_button')))
        for left, right in zip(integrated.dataframe, standalone.dataframe):
            pd.testing.assert_frame_equal(left.value, right.value)
        for chart in integrated.get('plotly_chart'):
            layout = json.loads(chart.proto.spec)['layout']
            self.assertEqual('#0a0e27', layout['paper_bgcolor'])
            self.assertEqual('#1e2139', layout['plot_bgcolor'])

    def test_05_form_recomputes_and_widget_rerun_keeps_route(self):
        at = self.app('risk')
        widget(at.number_input, 'Long horizon (trading days)').set_value(5)
        widget(at.number_input, 'Monte Carlo simulations').set_value(1000)
        widget(at.slider, 'Estimation window (trading days)').set_value(500)
        widget(at.radio, 'Monte Carlo return model').set_value('Student-t distribution (5 degrees of freedom)')
        widget(at.toggle, 'Include Monte Carlo VaR in the backtest').set_value(False)
        at.button(key='FormSubmitter:settings-Run analysis').click().run()
        self.assert_healthy(at)
        config = at.session_state.config
        self.assertEqual((5,1000,500,'t',False), (config['horizon'],config['n_sims'],config['window'],config['dist'],config['bt_mc']))
        at.slider(key='rho').set_value(0.8).run()
        self.assert_healthy(at)
        self.assertEqual('risk', at.query_params['page'][0])
        self.assertEqual(5, at.session_state.config['horizon'])

    def test_06_invalid_settings_keep_last_valid_results(self):
        at = self.app('risk')
        for label in ['95%','99%']:
            widget(at.checkbox, label).uncheck()
        at.button(key='FormSubmitter:settings-Run analysis').click().run()
        self.assertEqual([], [x.message for x in at.exception])
        self.assertIn('Tick at least one confidence level.', [x.value for x in at.error])
        self.assertEqual((0.95,0.99), at.session_state.config['conf'])
        self.assertEqual(TABS, [tab.label for tab in at.tabs])

    def test_07_existing_route_and_unknown_route(self):
        existing = self.app('monte-carlo-gambling')
        self.assert_healthy(existing)
        self.assertEqual('Monte Carlo Gambling Simulator', existing.title[0].value)
        unknown = self.app('not-a-project')
        self.assert_healthy(unknown)
        self.assertEqual('Home', unknown.session_state.current_page)


if __name__ == '__main__':
    unittest.main()
