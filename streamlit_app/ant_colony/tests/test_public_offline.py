from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


class PublicOfflineTests(unittest.TestCase):
    def test_initial_render_cache_clear_and_relaunch_without_network(self):
        # Tous les accès d'acquisition échoueraient si la page les appelait.
        with patch('streamlit_app.ant_colony.core.fetch_market_data', side_effect=AssertionError('Network forbidden')), \
             patch('yfinance.download', side_effect=AssertionError('Network forbidden')), \
             patch('requests.sessions.Session.request', side_effect=AssertionError('Network forbidden')):
            st.cache_data.clear()
            app = AppTest.from_file(str(ROOT / 'streamlit_app/app.py'), default_timeout=60)
            app.query_params['page'] = 'ant-colony-portfolio'
            app.run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertTrue(any('Données figées' in element.value for element in app.info))
            labels = [button.label for button in app.button]
            self.assertFalse(any('Actualiser les données' in label for label in labels))
            self.assertNotIn('Recharger les donnees', labels)
            self.assertEqual(len(app.get('file_uploader')), 0)
            self.assertTrue(app.session_state['aco_result']['result']['feasible'])
            metrics = {metric.label: metric.value for metric in app.metric}
            self.assertEqual(metrics['Ratio de Sharpe'], '0.63')
            self.assertEqual(metrics['Ratio de Sortino'], '0.89')
            self.assertTrue(any('taux sans risque : 0% par an' in caption.value for caption in app.caption))
            self.assertTrue(any('1\u202f230 clôtures communes' in caption.value for caption in app.caption))
            self.assertFalse(any('1,230' in caption.value or '3,200' in caption.value for caption in app.caption))
            # Effacer le cache ne doit pas transformer la lecture locale en fetch.
            st.cache_data.clear()
            app.slider[0].set_value(20)
            app.button(key='FormSubmitter:aco_parameters-Lancer la colonie').click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(app.session_state['aco_result']['result']['config']['n_ants'], 20)


if __name__ == '__main__':
    unittest.main()
