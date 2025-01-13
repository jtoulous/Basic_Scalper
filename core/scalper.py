import logging
import argparse
import pandas as pd
from datetime import datetime, timedelta

from utils.agent import Agent
from utils.arguments import ActiveCryptos
from utils.tools import TakeProfit, StopLoss
from utils.binance import DownloadData

def Parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('-risk', type=float, default=0.01, help='percentage of capital at risk on each trades')
    parser.add_argument('-sl', type=float, default=0.001, help='to calculate stop loss, use the same as training')
    parser.add_argument('-tp', type=float, default=0.002, help='to calculate take profit, use the same as training')
    args = parser.parse_args()
    return args



def LoadAgents():
    agents = {}
    for crypto in ActiveCryptos():
        agents[crypto] = Agent(crypto, load=True)
    return agents


#def LoadPreprocessPipelines(features):
#    preprocesses = {}
#    for crypto in ActiveCryptos():
#        preprocesses[crypto] = Pipeline([
#            ('RSI', RSI(14)),
#            ('BLG', BLG(20, 2)),
#            ('Scaler', Scaler(features, crypto, 'predict'))
#        ])
#    return preprocesses


def InitDataset(crypto):
    timeframe = '1m'
    oldest_timestamp = datetime.now() - timedelta(minutes=21)
    newest_timestamp = datetime.now()

    raw_data = DownloadData(crypto, timeframe=timeframe, interval=[oldest_timestamp, newest_timestamp], limit=100)
    dataframe = pd.DataFrame(raw_data, columns=['DATETIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'timestamp_end', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    dataframe = dataframe[['DATETIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']]

    dataframe['DATETIME'] = pd.to_datetime(dataframe['DATETIME'], unit='ms')
    dataframe['OPEN'] = dataframe['OPEN'].astype(float)
    dataframe['HIGH'] = dataframe['HIGH'].astype(float)
    dataframe['LOW'] = dataframe['LOW'].astype(float)
    dataframe['CLOSE'] = dataframe['CLOSE'].astype(float)
    dataframe['VOLUME'] = dataframe['VOLUME'].astype(float)

    return dataframe


#def UpdateDataset(crypto, dataset):


if __name__ == '__main__':
    try:
        args = Parsing()
#        agents = LoadAgents()
        datasets = {}


        preprocess = Pipeline([
            ('RSI', RSI(14)),
            ('BLG', BLG(20, 2)),
        ])   ### FAIRE LE SCALING JUSTE AVANT LA PREDICTION


        for crypto in ActiveCryptos():
            datasets[crypto] = InitDataset(crypto)
        last_entry = datasets['BTCUSDT'].iloc[-1]['DATETIME']

        while True:
            if last_entry.minute != datetime.now().minute:
                for crypto in ActiveCryptos():
                    datasets[crypto] = UpdateDataset(crypto, datasets[crypto])
                    X = preprocess.transform(datasets[crypto].copy())
                    X = X.iloc[-1]
                    prediction = agent[crypto].predict(X)
                    if prediction == 0:
                        take_profit = TakeProfit(X, args.tp)
                        stop_loss = StopLoss(X, args.sl)
#
                        Market_trade(crypto, args.risk, stop_loss)  #to do, mettre une protec si il y a plus de capital dispo
                        OCO_trade(crypto, args.risk, take_profit, stop_loss)#to do
                last_entry = datetime.now()

    except Exception as error:
        print(error)