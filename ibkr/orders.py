"""
IBKR Order Manager
==================
Calculates rebalancing orders and executes them via IBKR.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

from ib_insync import IB, Stock, MarketOrder, LimitOrder
import pandas as pd
import os

from .connection import IBKRConnection
from .config import (
    REBALANCE_THRESHOLD, MAX_ORDER_VALUE_USD, MIN_ORDER_SIZE_USD,
    ORDER_TYPE, CONTRACT_DETAILS, HOST, CURRENT_PORT, CLIENT_ID,
    CONNECTION_TIMEOUT, ETF_MAPPING, ASSETS_DATA_PATH, FOREX_DATA_PATH, ACCOUNT_ID
)

logger = logging.getLogger(__name__)


@dataclass
class RebalanceOrder:
    """Represents a rebalancing order to execute."""
    asset_name: str
    symbol: str
    action: str  # 'BUY' or 'SELL'
    shares: int
    estimated_value: float
    current_weight: float
    target_weight: float
    weight_delta: float


class OrderManager:
    """
    Manages order calculation and execution for portfolio rebalancing.
    
    The rebalancing logic:
    1. Compare current weights to target weights
    2. Only rebalance if delta > threshold (default 2%)
    3. Execute SELL orders first to free up cash
    4. Then execute BUY orders
    """
    
    def __init__(self, host: str = HOST, port: int = CURRENT_PORT,
                 client_id: int = CLIENT_ID, timeout: int = CONNECTION_TIMEOUT,
                 account_id: str = ACCOUNT_ID):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.timeout = timeout
        self.account_id = account_id
        self.ib: Optional[IB] = None
    
    def connect(self) -> bool:
        """Connect to IBKR."""
        try:
            self.ib = IB()
            self.ib.connect(self.host, self.port, clientId=self.client_id, timeout=self.timeout)
            self.ib.reqMarketDataType(3)
            logger.info(f"OrderManager connected to {self.host}:{self.port} (using delayed market data)")
            return True
        except Exception as e:
            logger.error(f"OrderManager connection failed: {e}")
            self.ib = None
            return False
    
    def disconnect(self):
        """Disconnect from IBKR."""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            logger.info("OrderManager disconnected")
        self.ib = None
    
    def get_current_price(self, asset_name: str) -> Optional[float]:
        """
        Get current market price for an asset using IBKR delayed data.
        Uses IBKR API only to ensure ticker consistency with order execution.
        
        Args:
            asset_name: Internal asset name (e.g., 'SP500')
            
        Returns:
            Current price or None if unavailable.
        """
        ibkr_symbol = ETF_MAPPING.get(asset_name)
        
        if not ibkr_symbol:
            logger.error(f"No ETF mapping for {asset_name}")
            return None
        
        if not self.ib or not self.ib.isConnected():
            logger.error("Not connected to IBKR")
            return None
        
        try:
            # Create contract (same as used for orders)
            from ib_insync import Contract
            details = CONTRACT_DETAILS.get(ibkr_symbol, {})
            
            contract_kwargs = {
                'symbol': details.get('symbol', ibkr_symbol),
                'secType': details.get('secType', 'STK'),
                'exchange': details.get('exchange', 'SMART'),
                'currency': details.get('currency', 'EUR')
            }
            if 'primaryExchange' in details:
                contract_kwargs['primaryExchange'] = details['primaryExchange']
            if 'secIdType' in details:
                contract_kwargs['secIdType'] = details['secIdType']
                contract_kwargs['secId'] = details['secId']
                
            contract = Contract(**contract_kwargs)
            qualified = self.ib.qualifyContracts(contract)
            
            # Fallback: If SMART fails and we have a primaryExchange, try absolute exchange
            if (not qualified or not contract.conId) and 'primaryExchange' in details:
                logger.warning(f"SMART qualification failed for {ibkr_symbol}, trying {details['primaryExchange']}")
                contract.exchange = details['primaryExchange']
                qualified = self.ib.qualifyContracts(contract)

            if not qualified or not contract.conId:
                logger.error(f"❌ Contract not found for {asset_name} ({ibkr_symbol})")
                return None
            
            logger.info(f"Contract found: {ibkr_symbol} conId={contract.conId}")
            
            # Request delayed market data
            ticker = self.ib.reqMktData(contract, '', False, False)
            self.ib.sleep(3)  # Wait for delayed data
            
            # Try market price first
            price = ticker.marketPrice()
            self.ib.cancelMktData(contract)
            
            if price and price > 0 and not (price != price):  # Check for NaN
                logger.info(f"{asset_name} ({ibkr_symbol}): IBKR price = €{price:.2f}")
                return price
            
            # Try close price as fallback
            if ticker.close and ticker.close > 0:
                logger.info(f"{asset_name} ({ibkr_symbol}): IBKR close = €{ticker.close:.2f}")
                return ticker.close
                
            logger.warning(f"No price data for {asset_name} ({ibkr_symbol}) from IBKR")
            return self._get_fallback_price(asset_name)
            
        except Exception as e:
            logger.error(f"IBKR price fetch failed for {ibkr_symbol}: {e}")
            return self._get_fallback_price(asset_name)

    def _get_fallback_price(self, asset_name: str) -> Optional[float]:
        """Read last known price from local data files if IBKR API fails."""
        try:
            # Check if it's a forex pair
            if asset_name in ['USD_JPY', 'USD_EUR']:
                if not os.path.exists(FOREX_DATA_PATH):
                    return None
                df = pd.read_parquet(FOREX_DATA_PATH)
                if asset_name in df.columns:
                    price = float(df[asset_name].iloc[-1])
                    logger.info(f"Fallback selected for {asset_name}: {price:.4f} (Local Data)")
                    return price
            else:
                # Standard ETF/Asset
                if not os.path.exists(ASSETS_DATA_PATH):
                    return None
                df = pd.read_parquet(ASSETS_DATA_PATH)
                if asset_name in df.columns:
                    price = float(df[asset_name].iloc[-1])
                    logger.info(f"Fallback selected for {asset_name}: {price:.4f} (Local Data)")
                    return price
            return None
        except Exception as e:
            logger.error(f"Fallback price fetch failed for {asset_name}: {e}")
            return None
    
    def calculate_rebalance_orders(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float,
        threshold: float = REBALANCE_THRESHOLD,
        base_currency: str = 'EUR'
    ) -> List[RebalanceOrder]:
        """
        Calculate orders needed to rebalance portfolio.
        
        Args:
            current_weights: Current portfolio weights by asset
            target_weights: Target portfolio weights by asset
            portfolio_value: Total portfolio value in USD
            threshold: Minimum weight difference to trigger rebalance (default 2%)
            
        Returns:
            List of RebalanceOrder objects (SELL orders first, then BUY)
        """
        if not self.ib or not self.ib.isConnected():
            raise ConnectionError("Not connected to IBKR")
        
        orders = []
        
        for asset_name, target_weight in target_weights.items():
            current_weight = current_weights.get(asset_name, 0.0)
            weight_delta = target_weight - current_weight
            
            # Skip if delta is below threshold
            if abs(weight_delta) < threshold:
                logger.debug(f"{asset_name}: delta {weight_delta:.2%} < threshold {threshold:.2%}, skipping")
                continue
            
            # Calculate order value
            order_value = abs(weight_delta) * portfolio_value
            
            # Skip dust orders
            if order_value < MIN_ORDER_SIZE_USD:
                logger.debug(f"{asset_name}: order value ${order_value:.2f} < min ${MIN_ORDER_SIZE_USD}, skipping")
                continue
            
            # Cap order value for safety
            if order_value > MAX_ORDER_VALUE_USD:
                logger.warning(f"{asset_name}: order value ${order_value:.2f} exceeds max ${MAX_ORDER_VALUE_USD}, capping")
                order_value = MAX_ORDER_VALUE_USD
            
            # Get contract details to check type
            mapping_symbol = ETF_MAPPING.get(asset_name)
            if not mapping_symbol:
                logger.error(f"No symbol mapping for asset: {asset_name}")
                continue
                
            details = CONTRACT_DETAILS.get(mapping_symbol, {})
            is_forex = details.get('secType') == 'CASH'
            real_symbol = details.get('symbol', mapping_symbol) # e.g., 'EUR' for USD_EUR
            
            price = self.get_current_price(asset_name)
            if not price:
                logger.error(f"Could not get price for {asset_name}, skipping")
                continue
            
            # Calculate shares
            if is_forex:
                # For Forex (CASH), the quantity is in the BASE currency of the pair (the 'real_symbol')
                # USD.JPY -> quantity is in USD (symbol=USD)
                # EUR.USD -> quantity is in EUR (symbol=EUR)
                
                if real_symbol == base_currency:
                    # Account base matches contract base (e.g., USD account, USD.JPY pair)
                    # We want to buy/sell X USD. Quantity IS X.
                    shares = int(order_value)
                else:
                    # Account base is DIFFERENT from contract base (e.g., USD account, EUR.USD pair)
                    # To get X USD exposure by trading EUR, we need X / price_of_EURUSD shares.
                    shares = int(order_value / price)
            else:
                # For Stocks/ETFs
                shares = int(order_value / price)
                
            if shares < 1:
                logger.debug(f"{asset_name}: calculated shares < 1, skipping")
                continue
            
            # Directions for Forex inversion
            action = 'BUY' if weight_delta > 0 else 'SELL'
            
            if is_forex and asset_name.startswith('USD_') and not real_symbol.startswith('USD'):
                # Inversion: Asset objective is USD, but contract base is NOT USD (e.g., EUR.USD)
                # To get USD (BUY USD_EUR), we must SELL the base (EUR)
                action = 'SELL' if weight_delta > 0 else 'BUY'
                logger.info(f"Forex Action Inversion for {asset_name} ({real_symbol}): {weight_delta:+.2%} -> {action}")

            order = RebalanceOrder(
                asset_name=asset_name,
                symbol=real_symbol,
                action=action,
                shares=shares,
                estimated_value=shares * price if real_symbol != base_currency else shares,
                current_weight=current_weight,
                target_weight=target_weight,
                weight_delta=weight_delta
            )
            orders.append(order)
            
            logger.info(
                f"Order: {action} {shares} {symbol} "
                f"(~${order.estimated_value:.2f}) "
                f"[{current_weight:.1%} → {target_weight:.1%}]"
            )
        
        # Sort: SELL orders first (to free up cash), then BUY
        orders.sort(key=lambda x: (0 if x.action == 'SELL' else 1, -x.estimated_value))
        
        return orders
    
    def execute_orders(
        self,
        orders: List[RebalanceOrder],
        dry_run: bool = False
    ) -> Dict[str, any]:
        """
        Execute a list of orders.
        
        Args:
            orders: List of RebalanceOrder objects
            dry_run: If True, only log orders without executing
            
        Returns:
            Execution summary dict
        """
        if not self.ib or not self.ib.isConnected():
            raise ConnectionError("Not connected to IBKR")
        
        results = {
            'executed': [],
            'failed': [],
            'skipped': [],
            'dry_run': dry_run
        }
        
        for order in orders:
            try:
                if dry_run:
                    logger.info(f"[DRY RUN] Would {order.action} {order.shares} {order.symbol}")
                    results['skipped'].append({
                        'symbol': order.symbol,
                        'action': order.action,
                        'shares': order.shares,
                        'reason': 'dry_run'
                    })
                    continue
                
                from ib_insync import Contract
                details = CONTRACT_DETAILS.get(order.symbol, {})
                
                contract_kwargs = {
                    'symbol': details.get('symbol', order.symbol),
                    'secType': details.get('secType', 'STK'),
                    'exchange': details.get('exchange', 'SMART'),
                    'currency': details.get('currency', 'EUR')
                }
                if 'primaryExchange' in details:
                    contract_kwargs['primaryExchange'] = details['primaryExchange']
                if 'secIdType' in details:
                    contract_kwargs['secIdType'] = details['secIdType']
                    contract_kwargs['secId'] = details['secId']
                    
                contract = Contract(**contract_kwargs)
                qualified = self.ib.qualifyContracts(contract)
                
                # Fallback: If SMART fails and we have a primaryExchange, try absolute exchange
                if (not qualified or not contract.conId) and 'primaryExchange' in details:
                    logger.warning(f"SMART qualification failed for {order.symbol}, trying {details['primaryExchange']}")
                    contract.exchange = details['primaryExchange']
                    qualified = self.ib.qualifyContracts(contract)

                # Check if contract was found
                if not qualified or not contract.conId:
                    logger.error(f"❌ Contract not found for {order.symbol} - skipping order")
                    results['failed'].append({
                        'symbol': order.symbol,
                        'action': order.action,
                        'shares': order.shares,
                        'reason': 'contract_not_found'
                    })
                    continue
                
                logger.info(f"Contract qualified: {order.symbol} conId={contract.conId}")
                
                # Create order
                if ORDER_TYPE == 'MKT':
                    ib_order = MarketOrder(order.action, order.shares, tif='DAY', account=self.account_id or '')
                else:
                    # For limit orders, use current price
                    price = self.get_current_price(order.asset_name)
                    ib_order = LimitOrder(order.action, order.shares, price, tif='DAY', account=self.account_id or '')
                
                # Place order
                trade = self.ib.placeOrder(contract, ib_order)
                self.ib.sleep(1)  # Wait for order to be processed
                
                logger.info(
                    f"✅ Executed: {order.action} {order.shares} {order.symbol} "
                    f"- Status: {trade.orderStatus.status}"
                )
                
                results['executed'].append({
                    'symbol': order.symbol,
                    'action': order.action,
                    'shares': order.shares,
                    'status': trade.orderStatus.status,
                    'order_id': trade.order.orderId
                })
                
            except Exception as e:
                logger.error(f"❌ Failed to execute {order.action} {order.shares} {order.symbol}: {e}")
                results['failed'].append({
                    'symbol': order.symbol,
                    'action': order.action,
                    'shares': order.shares,
                    'error': str(e)
                })
        
        return results


def test_order_calculation():
    """Test order calculation without executing."""
    # Mock data for testing
    current_weights = {
        'SP500': 0.35,
        'GOLD_OZ_USD': 0.25,
        'TREASURY_10Y': 0.40,
        'SmallCAP': 0.0,
        'US_REIT_VNQ': 0.0,
        'OBLIGATION': 0.0,
        'NASDAQ_100': 0.0,
        'COMMODITIES': 0.0
    }
    
    # Target: Q1 allocation
    target_weights = {
        'SP500': 0.30,
        'NASDAQ_100': 0.40,
        'SmallCAP': 0.30,
        'GOLD_OZ_USD': 0.0,
        'TREASURY_10Y': 0.0,
        'US_REIT_VNQ': 0.0,
        'OBLIGATION': 0.0,
        'COMMODITIES': 0.0
    }
    
    portfolio_value = 10000.0
    
    om = OrderManager()
    if om.connect():
        try:
            orders = om.calculate_rebalance_orders(
                current_weights, target_weights, portfolio_value
            )
            print(f"\n📋 Orders to execute ({len(orders)} total):")
            for o in orders:
                print(f"  {o.action} {o.shares} {o.symbol} (~${o.estimated_value:.2f})")
        finally:
            om.disconnect()
    else:
        print("❌ Failed to connect to IBKR")


if __name__ == "__main__":
    test_order_calculation()
