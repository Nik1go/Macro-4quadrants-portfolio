import sys
import logging
import asyncio

# Create dummy event loop if not present
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from ib_insync import IB, Contract
from ibkr.config import ETF_MAPPING, CONTRACT_DETAILS, HOST, CURRENT_PORT, CLIENT_ID

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def test_tickers():
    ib = IB()
    try:
        # Use a high client ID to avoid conflicts with other scripts
        ib.connect(HOST, CURRENT_PORT, clientId=999)
        logging.info("Successfully connected to IBKR Gateway")
    except Exception as e:
        logging.error(f"Could not connect: {e}")
        return

    # Request delayed data if live is not available
    ib.reqMarketDataType(3)

    print("\n" + "="*50)
    print("Testing IBKR Contract Qualifications")
    print("="*50)

    for asset_name, ibkr_symbol in ETF_MAPPING.items():
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
        
        # Qualify Contract
        qualified_list = ib.qualifyContracts(contract)
        
        if not qualified_list and 'primaryExchange' in details:
            contract.exchange = details['primaryExchange']
            qualified_list = ib.qualifyContracts(contract)

        if not qualified_list:
            print(f"❌ FAILED to qualify: {asset_name} -> {contract_kwargs}")
            continue
            
        qualified_contract = qualified_list[0]
        con_id = qualified_contract.conId
        symbol = qualified_contract.symbol
        exch = qualified_contract.exchange
        
        print(f"✅ QUALIFIED: {asset_name:12s} | ID: {con_id} | {symbol} on {exch}")
        
        try:
            # Request price
            ticker = ib.reqMktData(qualified_contract, '', False, False)
            ib.sleep(2)
            
            price = ticker.marketPrice()
            if price != price or price == 0:  # NaN or 0
                price = ticker.close
                
            if price and price > 0 and price == price:
                print(f"   ↳ Price verified: {price} {qualified_contract.currency}")
            else:
                print(f"   ↳ Contract valid, but could not retrieve live/delayed price.")
                
            ib.cancelMktData(qualified_contract)
            
        except Exception as e:
            print(f"   ↳ Error fetching price: {e}")

    print("="*50 + "\n")
    ib.disconnect()

if __name__ == "__main__":
    test_tickers()
