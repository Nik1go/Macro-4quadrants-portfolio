"""
IBKR Portfolio Manager
======================
Reads current positions and calculates portfolio weights from IBKR.
"""

from typing import Dict, Optional
import logging

from ib_insync import IB, Stock

from .connection import IBKRConnection
from .config import ETF_MAPPING, HOST, CURRENT_PORT, CLIENT_ID, CONNECTION_TIMEOUT, ACCOUNT_ID

logger = logging.getLogger(__name__)


class PortfolioManager:
    """
    Manages portfolio reading from IBKR.
    
    Usage:
        pm = PortfolioManager()
        if pm.connect():
            weights = pm.get_current_weights()
            value = pm.get_portfolio_value()
            pm.disconnect()
    """
    
    def __init__(self, host: str = HOST, port: int = CURRENT_PORT, 
                 client_id: int = CLIENT_ID, timeout: int = CONNECTION_TIMEOUT,
                 account_id: str = ACCOUNT_ID):
        import random
        self.host = host
        self.port = port
        self.client_id = client_id if client_id != 1 else random.randint(1000, 9999)
        self.timeout = timeout
        self.account_id = account_id
        self.ib: Optional[IB] = None
        
        # Build symbol mapping: IBKR symbol -> internal name
        from .config import CONTRACT_DETAILS
        self.symbol_to_asset = {v: k for k, v in ETF_MAPPING.items()}
        # Add overrides from CONTRACT_DETAILS (crucial for CFDs: real symbol 'EUR' -> asset 'USD_EUR')
        for asset, ibkr_id in ETF_MAPPING.items():
            if ibkr_id in CONTRACT_DETAILS and 'symbol' in CONTRACT_DETAILS[ibkr_id]:
                real_symbol = CONTRACT_DETAILS[ibkr_id]['symbol']
                self.symbol_to_asset[real_symbol] = asset
                logger.debug(f"Mapping real symbol {real_symbol} to asset {asset}")
        
    def connect(self) -> bool:
        """Connect to IBKR."""
        try:
            self.ib = IB()
            self.ib.connect(self.host, self.port, clientId=self.client_id, timeout=self.timeout)
            logger.info(f"PortfolioManager connected to {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"PortfolioManager connection failed: {e}")
            self.ib = None
            return False
    
    def disconnect(self):
        """Disconnect from IBKR."""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            logger.info("PortfolioManager disconnected")
        self.ib = None
    
    def get_positions(self) -> Dict[str, Dict]:
        """
        Get actual positions from IBKR (real portfolio only).
        """
        if not self.ib or not self.ib.isConnected():
            raise ConnectionError("Not connected to IBKR")
        
        positions = {}
        self.ib.waitOnUpdate(timeout=1.0)
        
        portfolio_items = self.ib.portfolio()
        if self.account_id:
            portfolio_items = [item for item in portfolio_items if item.account == self.account_id]
        
        for item in portfolio_items:
            symbol = item.contract.symbol
            if symbol in self.symbol_to_asset:
                asset_name = self.symbol_to_asset[symbol]
                positions[asset_name] = {
                    'symbol': symbol,
                    'shares': item.position,
                    'avg_cost': item.averageCost,
                    'market_value': item.marketValue,
                    'unrealized_pnl': item.unrealizedPNL,
                    'currency': item.contract.currency
                }
        
        return positions
    
    def get_portfolio_value(self) -> float:
        """
        Get total portfolio value (cash + positions).
        
        Returns:
            Total portfolio value in account base currency.
        """
        if not self.ib or not self.ib.isConnected():
            raise ConnectionError("Not connected to IBKR")
        
        # Request account summary (optionally filtered by account)
        account_values = self.ib.accountSummary(account=self.account_id or '')
        
        # Try to find NetLiquidation in various currencies (EUR first, then USD, then BASE)
        for currency in ['EUR', 'USD', 'BASE']:
            for av in account_values:
                if self.account_id and av.account != self.account_id:
                    continue
                if av.tag == 'NetLiquidation' and av.currency == currency:
                    value = float(av.value)
                    if value > 0:
                        logger.info(f"Portfolio value: {value:.2f} {currency}")
                        return value
        
        return 0.0

    def get_base_currency(self) -> str:
        """
        Detect the account base currency.
        
        Returns:
            Currency code (e.g., 'EUR', 'USD').
        """
        if not self.ib or not self.ib.isConnected():
            raise ConnectionError("Not connected to IBKR")
            
        account_values = self.ib.accountSummary(account=self.account_id or '')
        
        # Try to find NetLiquidation or TotalCashValue currency
        for av in account_values:
            if self.account_id and av.account != self.account_id:
                continue
            if av.tag == 'NetLiquidation' and float(av.value) > 0:
                return av.currency
                
        # Fallback to EUR or USD if found, else default to EUR
        for currency in ['EUR', 'USD']:
            if any(av.currency == currency for av in account_values):
                return currency
                
        return 'EUR'

    def get_exchange_rate(self, from_currency: str, to_currency: str) -> float:
        """Get exchange rate between two currencies."""
        if from_currency == to_currency:
            return 1.0
            
        try:
            # Check local fallback data first (fast and reliable)
            from .orders import OrderManager
            om = OrderManager()
            # If we want e.g. JPY to EUR, we might have USD_JPY and USD_EUR
            # Rate = (1/USD_JPY) * USD_EUR
            
            usd_from = om._get_fallback_price(f'USD_{from_currency}')
            usd_to = om._get_fallback_price(f'USD_{to_currency}')
            
            if from_currency == 'USD' and usd_to:
                return usd_to # USD to EUR
            if to_currency == 'USD' and usd_from:
                return 1.0 / usd_from # EUR to USD
            if usd_from and usd_to:
                # e.g. JPY to EUR: (USD/JPY)^-1 * (USD/EUR) = (JPY/USD)^-1 * (EUR/USD) ? No.
                # USD_JPY is JPY per USD. USD_EUR is EUR per USD.
                # So 1 USD = X JPY and 1 USD = Y EUR.
                # Thus X JPY = Y EUR => 1 JPY = Y/X EUR.
                return usd_to / usd_from
                
            return 1.0
        except Exception as e:
            logger.warning(f"Could not get exchange rate {from_currency}->{to_currency}: {e}")
            return 1.0
    
    def get_current_weights(self) -> Dict[str, float]:
        """
        Calculate current portfolio weights (real positions only).
        """
        if not self.ib or not self.ib.isConnected():
            raise ConnectionError("Not connected to IBKR")
        
        positions = self.get_positions()
        total_value = self.get_portfolio_value()
        
        if total_value <= 0:
            logger.warning("Portfolio value is zero or negative")
            return {asset: 0.0 for asset in ETF_MAPPING.keys()}
        
        weights = {}
        base_currency = self.get_base_currency()
        
        # Calculate weight for each asset
        for asset_name in ETF_MAPPING.keys():
            if asset_name in positions:
                pos = positions[asset_name]
                market_value = pos['market_value']
                currency = pos.get('currency', base_currency)
                
                # Convert to base currency if needed
                if currency != base_currency:
                    rate = self.get_exchange_rate(currency, base_currency)
                    market_value_in_base = market_value * rate
                    logger.debug(f"Converted {asset_name} value: {market_value:.2f} {currency} -> {market_value_in_base:.2f} {base_currency}")
                    market_value = market_value_in_base
                
                weights[asset_name] = market_value / total_value
            else:
                weights[asset_name] = 0.0
        
        logger.info(f"Current weights (converted to {base_currency}): {weights}")
        return weights
    
    def get_cash_balance(self) -> float:
        """Get available cash balance in account base currency."""
        if not self.ib or not self.ib.isConnected():
            raise ConnectionError("Not connected to IBKR")
        
        account_values = self.ib.accountSummary()
        
        # Try EUR first (for European accounts), then USD
        for currency in ['EUR', 'USD']:
            for av in account_values:
                if av.tag == 'TotalCashValue' and av.currency == currency:
                    value = float(av.value)
                    if value > 0:
                        return value
        
        return 0.0



def test_portfolio():
    """Quick test to read portfolio from IBKR."""
    pm = PortfolioManager()
    
    if pm.connect():
        try:
            print(f"📊 Portfolio Value: ${pm.get_portfolio_value():,.2f}")
            print(f"💵 Cash Balance: ${pm.get_cash_balance():,.2f}")
            print(f"📈 Positions: {pm.get_positions()}")
            print(f"⚖️ Weights: {pm.get_current_weights()}")
        finally:
            pm.disconnect()
    else:
        print("❌ Failed to connect to IBKR")


if __name__ == "__main__":
    test_portfolio()
