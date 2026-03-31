"""
Black-Scholes model implementation for digital options (Cash-or-Nothing).
"""
from typing import Union, Optional
import numpy as np
from scipy.stats import norm


class BlackScholesDigitalOption:
    """
    Black-Scholes model for digital options (Cash-or-Nothing).
    
    This class provides methods to calculate the theoretical price (probability)
    of digital options using the Black-Scholes model.
    """
    
    @staticmethod
    def calculate_d1(
        S: float, 
        K: float, 
        T: float, 
        r: float, 
        sigma: float
    ) -> float:
        """
        Calculate d1 in the Black-Scholes formula.
        
        Args:
            S: Current spot price of the underlying asset
            K: Strike price
            T: Time to maturity in years
            r: Risk-free interest rate (annual)
            sigma: Volatility of the underlying asset (annual)
            
        Returns:
            float: The d1 value in the Black-Scholes formula
        """
        if T <= 0 or sigma <= 0:
            return float('inf') if S > K else float('-inf')
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def calculate_d2(
        S: float, 
        K: float, 
        T: float, 
        r: float, 
        sigma: float
    ) -> float:
        """
        Calculate d2 in the Black-Scholes formula.
        
        Args:
            S: Current spot price of the underlying asset
            K: Strike price
            T: Time to maturity in years
            r: Risk-free interest rate (annual)
            sigma: Volatility of the underlying asset (annual)
            
        Returns:
            float: The d2 value in the Black-Scholes formula
        """
        if T <= 0 or sigma <= 0:
            return float('inf') if S > K else float('-inf')
        return (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def digital_call_price(
        S: float, 
        K: float, 
        T: float, 
        r: float, 
        sigma: float
    ) -> float:
        """
        Calculate the price of a digital Call option (Cash-or-Nothing).
        Returns the probability that S > K at maturity.
        
        Args:
            S: Current spot price of the underlying asset
            K: Strike price
            T: Time to maturity in years
            r: Risk-free interest rate (annual)
            sigma: Volatility of the underlying asset (annual)
            
        Returns:
            float: Probability between 0 and 1
        """
        if T <= 0:
            return 1.0 if S > K else 0.0
        
        d2 = BlackScholesDigitalOption.calculate_d2(S, K, T, r, sigma)
        return norm.cdf(d2)
    
    @staticmethod
    def digital_put_price(
        S: float, 
        K: float, 
        T: float, 
        r: float, 
        sigma: float
    ) -> float:
        """
        Calculate the price of a digital Put option (Cash-or-Nothing).
        Returns the probability that S < K at maturity.
        
        Args:
            S: Current spot price of the underlying asset
            K: Strike price
            T: Time to maturity in years
            r: Risk-free interest rate (annual)
            sigma: Volatility of the underlying asset (annual)
            
        Returns:
            float: Probability between 0 and 1
        """
        if T <= 0:
            return 1.0 if S < K else 0.0
        
        d2 = BlackScholesDigitalOption.calculate_d2(S, K, T, r, sigma)
        return norm.cdf(-d2)