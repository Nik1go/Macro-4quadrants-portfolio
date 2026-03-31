"""
Kelly Criterion implementation for optimal position sizing.
"""
from typing import Union, Optional


class KellyCriterion:
    """
    Implementation of the Kelly Criterion for optimal position sizing.
    
    The Kelly Criterion calculates the optimal fraction of capital to allocate
    to a bet with a positive expected value, maximizing the expected logarithm
    of wealth.
    """
    
    @staticmethod
    def calculate_fraction(
        win_prob: float, 
        win_loss_ratio: float
    ) -> float:
        """
        Calculate the optimal fraction of capital to risk.
        
        Args:
            win_prob: Probability of winning (between 0 and 1)
            win_loss_ratio: Ratio of potential gain to potential loss
            
        Returns:
            float: Optimal fraction of capital to risk (between 0 and 1)
        """
        if win_prob <= 0 or win_loss_ratio <= 0:
            return 0
            
        # Classic Kelly formula: f* = p - (1-p)/r
        # where p = win probability, r = win/loss ratio
        kelly_fraction = win_prob - (1 - win_prob) / win_loss_ratio
        
        # Limit to positive values
        return max(0, kelly_fraction)
    
    @staticmethod
    def calculate_position_size(
        capital: float, 
        kelly_fraction: float, 
        max_risk_pct: float = 0.05, 
        half_kelly: bool = True
    ) -> float:
        """
        Calculate the position size based on available capital.
        
        Args:
            capital: Total available capital
            kelly_fraction: Kelly fraction calculated
            max_risk_pct: Maximum risk per trade (as a decimal)
            half_kelly: If True, use Half-Kelly (more conservative)
            
        Returns:
            float: Recommended position size
        """
        if capital <= 0 or kelly_fraction <= 0:
            return 0.0
            
        # Apply Half-Kelly for more conservative sizing
        if half_kelly:
            kelly_fraction = kelly_fraction / 2
            
        # Limit the maximum risk per trade
        risk_fraction = min(kelly_fraction, max_risk_pct)
        
        return capital * risk_fraction