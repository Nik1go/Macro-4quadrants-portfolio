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
        import random
        self.host = host
        self.port = port
        # Use random client ID if default is used to prevent connection blocking
        self.client_id = client_id if client_id != 1 else random.randint(1000, 9999)
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
            
            # Skip IBKR market data request for Forex and go straight to fallback
            # This prevents Error 10089 (Requires Subscription) and Error 300
            if contract.secType == 'CASH':
                logger.warning(f"No price data for {asset_name} ({ibkr_symbol}) from IBKR")
                return self._get_fallback_price(asset_name)
            
            # Request delayed market data
            ticker = self.ib.reqMktData(contract, '', False, False)
            self.ib.sleep(3)  # Wait for delayed data
            
            # Try market price first
            price = ticker.marketPrice()
            
            # Only cancel if we actually got a valid ticker ID to prevent Error 300
            # though skipping CASH should resolve the main cause.
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
        """
        if not self.ib or not self.ib.isConnected():
            raise ConnectionError("Not connected to IBKR")
        
        from .portfolio import PortfolioManager
        pm = PortfolioManager(account_id=self.account_id)
        if not pm.connect():
             raise ConnectionError("Could not connect PortfolioManager for rebalance calculation")
        
        try:
            current_positions = pm.get_positions()
        finally:
            pm.disconnect()
            
        orders = []
        
        all_assets = set(target_weights.keys()).union(ETF_MAPPING.keys()).union(current_positions.keys())
        
        for asset_name in all_assets:
            target_weight = target_weights.get(asset_name, 0.0)
            current_weight = current_weights.get(asset_name, 0.0)
            target_value_in_base = target_weight * portfolio_value
            
            mapping_symbol = ETF_MAPPING.get(asset_name)
            if not mapping_symbol:
                continue
                
            details = CONTRACT_DETAILS.get(mapping_symbol, {})
            is_forex = asset_name.startswith('USD_')
            real_symbol = details.get('symbol', mapping_symbol)
            
            price = self.get_current_price(asset_name)
            if not price:
                if target_weight == 0 and current_weight == 0:
                    continue
                logger.warning(f"Could not get price for {asset_name}, skipping.")
                continue
            
            # Calculate TARGET shares directly from TARGET weight
            asset_currency = details.get('symbol', mapping_symbol)
            if not is_forex:
                target_shares_magnitude = int(target_value_in_base / price)
                cross_rate_to_base = price
            else:
                if asset_currency == base_currency:
                    target_shares_magnitude = int(target_value_in_base)
                    cross_rate_to_base = 1.0
                else:
                    from .portfolio import PortfolioManager
                    pm = PortfolioManager()
                    rate_to_base = pm.get_exchange_rate(asset_currency, base_currency)
                    if rate_to_base > 0:
                        target_shares_magnitude = int(target_value_in_base / rate_to_base)
                        cross_rate_to_base = rate_to_base
                    else:
                        logger.error(f"❌ Could not determine exchange rate for {asset_currency} to {base_currency}")
                        continue

            # Apply inversion if needed
            if is_forex and asset_name.startswith('USD_') and not real_symbol.startswith('USD'):
                # Inverted asset: weight +0.18 means position -18%
                target_shares = -target_shares_magnitude
            else:
                target_shares = target_shares_magnitude
                
            # Current shares held
            current_shares = current_positions.get(asset_name, {}).get('shares', 0.0)
            
            # Delta to trade
            delta_shares = target_shares - current_shares
            
            # Check threshold over the resulting weight diff
            delta_weight = abs(delta_shares * cross_rate_to_base / portfolio_value)
            # Always close out completely if target is 0, ignoring the threshold
            if delta_weight < threshold and target_weight != 0.0:
                logger.debug(f"Skipping {asset_name}: delta {delta_weight:.1%} < threshold {threshold:.1%}")
                continue

            action = 'BUY' if delta_shares > 0 else 'SELL'
            final_shares = int(abs(delta_shares))

            if final_shares < 1:
                continue
                
            order_value_in_base = final_shares * cross_rate_to_base
            if order_value_in_base < MIN_ORDER_SIZE_USD:
                logger.debug(f"Skipping {asset_name}: value {order_value_in_base:.2f} < MIN_ORDER_SIZE {MIN_ORDER_SIZE_USD}")
                continue

            order = RebalanceOrder(
                asset_name=asset_name,
                symbol=real_symbol,
                action=action,
                shares=final_shares,
                estimated_value=order_value_in_base,
                current_weight=current_weight,
                target_weight=target_weight,
                weight_delta=delta_shares * cross_rate_to_base / portfolio_value
            )
            orders.append(order)
            
            logger.info(
                f"Order: {action} {final_shares} {real_symbol} "
                f"(~{order_value_in_base:.2f} {base_currency}) "
                f"[{current_weight:.1%} → {target_weight:.1%}]"
            )
        
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
            
        # FORCE CLEAN SLATE: Cancel everything and WAIT
        if not dry_run:
            logger.info("Cleaning up all pending orders before rebalance...")
            self.ib.reqAllOpenOrders()
            for _ in range(10): # Wait for sync
                self.ib.sleep(0.2)
                if self.ib.openTrades(): break
                
            all_trades = self.ib.openTrades()
            active_trades = [t for t in all_trades if not self.account_id or t.order.account == self.account_id]
            
            if active_trades:
                for trade in active_trades:
                    self.ib.cancelOrder(trade.order)
                
                # HARD WAIT for confirmation
                for i in range(15):
                    self.ib.sleep(0.5)
                    self.ib.reqOpenOrders() # Force update
                    still_open = [t for t in self.ib.openTrades() if not self.account_id or t.order.account == self.account_id]
                    if not still_open:
                        logger.info("Order book cleared and confirmed by IBKR.")
                        break
                    if i == 14:
                        logger.error("🛑 CRITICAL: Orders are still pending cancellation. Aborting execution to prevent double positions.")
                        return {'executed': [], 'failed': [{'asset': 'ALL', 'error': 'Cancellation Timeout'}], 'skipped': [], 'dry_run': False}
        
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
                        'asset': order.asset_name,
                        'symbol': order.symbol,
                        'action': order.action,
                        'shares': order.shares,
                        'reason': 'dry_run'
                    })
                    continue
                
                from ib_insync import Contract
                mapping_symbol = ETF_MAPPING.get(order.asset_name, order.symbol)
                details = CONTRACT_DETAILS.get(mapping_symbol, {})
                
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
                        'asset': order.asset_name,
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
                    'asset': order.asset_name,
                    'symbol': order.symbol,
                    'action': order.action,
                    'shares': order.shares,
                    'status': trade.orderStatus.status,
                    'order_id': trade.order.orderId
                })
                
            except Exception as e:
                logger.error(f"❌ Failed to execute {order.action} {order.shares} {order.symbol}: {e}")
                results['failed'].append({
                    'asset': order.asset_name,
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
