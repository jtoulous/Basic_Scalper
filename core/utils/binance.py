import os

from binance.client import Client
from utils.log import printLog
from utils.arguments import TranslateCryptoName



def TradeManager(crypto, open_price, stop_loss, take_profit, capital_risk):
    api_key = os.getenv('BINANCE_API')
    api_secret = os.getenv('BINANCE_SECRET')
    symbol = TranslateCryptoName(crypto)
#    api_key = os.getenv('TESTNET_API')
#    api_secret = os.getenv('TESTNET_SECRET')

    breakpoint()
    client = Client(api_key, api_secret)
    usdc_balance = GetAccountBalance(client, 'USDC')
    position_size = GetTradeSize(client, usdc_balance, capital_risk, open_price, stop_loss)

    breakpoint()
    buy_order = client.order_market_buy(
        symbol=symbol,
        quantity=position_size
    )
    printLog(f" Buy ==> {buy_order}")

    rounded_take_profit = round(take_profit, 4)
    rounded_stop_price = round(stop_loss * 1.02, 4)
    rounded_stop_limit_price = round(stop_loss, 4)

    oco_order = client.create_oco_order(
        symbol=symbol,
        side=Client.SIDE_SELL,
        quantity=buy_order['executedQty'],
        price=str(rounded_take_profit),
        stopPrice=str(rounded_stop_price),
        stopLimitPrice=str(rounded_stop_limit_price),
        stopLimitTimeInForce='GTC'
    )
    printLog(f' OCO order ==> {oco_order} ')


def GetTradeSize(client, usdc_balance, capital_risk, open_price, stop_loss):
    risk_amount = usdc_balance * (capital_risk / 100)
    stop_loss_distance = open_price - stop_loss
    position_size = risk_amount / (stop_loss_distance * open_price)
    return round(position_size, 1)

def GetAccountBalance(client, currency):
    balances = client.get_account()['balances']
    balance = next((float(b['free']) for b in balances if b['asset'] == currency), 0.0)
    return balance