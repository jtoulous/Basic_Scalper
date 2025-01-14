import os
import requests

from binance.client import Client
from utils.log import printLog

#from utils.arguments import TranslateCryptoName


#
#def TradeManager(crypto, open_price, stop_loss, take_profit, capital_risk):
#    api_key = os.getenv('BINANCE_API')
#    api_secret = os.getenv('BINANCE_SECRET')
##    api_key = os.getenv('TESTNET_API')
##    api_secret = os.getenv('TESTNET_SECRET')
#
#    breakpoint()
#    client = Client(api_key, api_secret)
#    usdc_balance = GetAccountBalance(client, 'USDC')
#    position_size = GetTradeSize(client, usdc_balance, capital_risk, open_price, stop_loss)
#
#    breakpoint()
#    buy_order = client.order_market_buy(
#        symbol=crypto,
#        quantity=position_size
#    )
#    printLog(f" Buy ==> {buy_order}")
#
#    rounded_take_profit = round(take_profit, 4)
#    rounded_stop_price = round(stop_loss * 1.02, 4)
#    rounded_stop_limit_price = round(stop_loss, 4)
#
#    oco_order = client.create_oco_order(
#        symbol=crypto,
#        side=Client.SIDE_SELL,
#        quantity=buy_order['executedQty'],
#        price=str(rounded_take_profit),
#        stopPrice=str(rounded_stop_price),
#        stopLimitPrice=str(rounded_stop_limit_price),
#        stopLimitTimeInForce='GTC'
#    )
#    printLog(f' OCO order ==> {oco_order} ')
#
#
#def GetTradeSize(client, usdc_balance, capital_risk, open_price, stop_loss):
#    risk_amount = usdc_balance * (capital_risk / 100)
#    stop_loss_distance = open_price - stop_loss
#    position_size = risk_amount / (stop_loss_distance * open_price)
#    return round(position_size, 1)
#
#def GetAccountBalance(client, currency):
#    balances = client.get_account()['balances']
#    balance = next((float(b['free']) for b in balances if b['asset'] == currency), 0.0)
#    return balance



def DownloadData(crypto, timeframe, timestamp=None, interval=None, limit=100):
    url = "https://api.binance.com/api/v1/klines"
    if timestamp is not None:
        start_time = int(timestamp.timestamp() * 1000)
        params = {
            "symbol": crypto,
            "interval": timeframe,
            "startTime": start_time,
            "limit": limit
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Erreur API: {response.status_code} - {response.text}")
            return None
    
    elif interval is not None:
#        start_timestamp = interval[0].to_pydatetime()
#        end_timestamp = interval[1].to_pydatetime()

        start_timestamp = int(interval[0].timestamp() * 1000)
        end_timestamp = int(interval[1].timestamp() * 1000)

#        start_timestamp = int(interval[0].value / 1e6)
#        end_timestamp = int(interval[1].value / 1e6)

        params = {
            "symbol": crypto,
            "interval": timeframe,
            "startTime": start_timestamp,
            "endTime": end_timestamp,
            "limit": limit
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Erreur API: {response.status_code} - {response.text}")
            return None



#def OCO_trade(crypto, risk, take_profit, stop_loss):
#    logging.info(f'Placing OCO on {crypto}')
#    
#    api_key = os.getenv('BINANCE_API')
#    api_secret = os.getenv('BINANCE_SECRET')
#    client = Client(api_key, api_secret)
#    
#    capital = GetAccountBalance(client, 'USDT')
#    risk_amount = capital * risk
#    entry_price = GetCurrentPrice(crypto)
#    stop_loss_distance = abs(entry_price - stop_loss)
#
#    if stop_loss_distance > 0:
#        position_size = risk_amount / stop_loss_distance
#    else:
#        logging.info(f'Too late for {crypto}')
#        return

    
        