"""
Performance metrics calculation and export for the Polymarket arbitrage system.
"""
import pandas as pd
import numpy as np
import json
from typing import Dict, Any, Optional
import os

from utils.config import Config
from utils.logger import Logger

logger = Logger().logger


class PerformanceMetrics:
    """
    Calculate and export performance metrics for the arbitrage bot.
    
    Provides methods to calculate key performance indicators such as
    Sharpe ratio, win rate, and maximum drawdown.
    """
    
    @staticmethod
    def calculate_metrics(trades_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate performance metrics from the trades DataFrame.
        
        Args:
            trades_df: DataFrame containing trade history
            
        Returns:
            Dict containing calculated metrics
        """
        if trades_df.empty:
            return {
                "sharpe_ratio": 0,
                "total_pnl": 0,
                "win_rate": 0,
                "max_drawdown": 0,
                "num_trades": 0
            }
        
        # Calculate total PnL
        total_pnl = trades_df['pnl'].sum()
        
        # Calculate win rate
        num_trades = len(trades_df)
        num_wins = len(trades_df[trades_df['pnl'] > 0])
        win_rate = num_wins / num_trades if num_trades > 0 else 0
        
        # Calculate Sharpe Ratio (if we have daily data)
        # Make sure timestamp is in datetime format
        if 'timestamp' in trades_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(trades_df['timestamp']):
                try:
                    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
                except Exception as e:
                    logger.error(f"Error converting timestamp to datetime: {e}")
        
        sharpe_ratio = 0
        try:
            daily_returns = trades_df.set_index('timestamp').resample('D')['pnl'].sum()
            if len(daily_returns) > 1:
                sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
        
        # Calculate Maximum Drawdown
        max_drawdown = 0
        try:
            cumulative_pnl = trades_df['pnl'].cumsum()
            if len(cumulative_pnl) > 0:
                running_max = cumulative_pnl.cummax()
                drawdown = (running_max - cumulative_pnl) / running_max.replace(0, np.nan).fillna(1)
                max_drawdown = drawdown.max() if len(drawdown) > 0 else 0
        except Exception as e:
            logger.error(f"Error calculating maximum drawdown: {e}")
        
        return {
            "sharpe_ratio": float(sharpe_ratio),
            "total_pnl": float(total_pnl),
            "win_rate": float(win_rate),
            "max_drawdown": float(max_drawdown),
            "num_trades": int(num_trades)
        }
    
    @staticmethod
    def export_metrics(trades_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate and export metrics to a JSON file.
        
        Args:
            trades_df: DataFrame containing trade history
            
        Returns:
            Dict containing calculated metrics
        """
        # Ensure data directory exists
        os.makedirs(os.path.dirname(Config.METRICS_JSON_PATH), exist_ok=True)
        
        metrics = PerformanceMetrics.calculate_metrics(trades_df)
        
        # Export to JSON
        try:
            with open(Config.METRICS_JSON_PATH, 'w') as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Metrics exported to {Config.METRICS_JSON_PATH}")
        except Exception as e:
            logger.error(f"Error exporting metrics to JSON: {e}")
        
        return metrics