import logging
import argparse
import pandas as pd
from datetime import datetime, timedelta

from sklearn.pipeline import Pipeline

from utils.agent import Agent
from utils.arguments import ActiveCryptos
from utils.tools import TakeProfit, StopLoss
from utils.binance import DownloadData
from utils.indicators import RSI, BLG


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
    quick_preprocess = Pipeline([
        ('RSI14', RSI(14)),
        ('BLG', BLG(20, 2))
    ])

    timeframe = '1m'
    oldest_timestamp = datetime.now() - timedelta(minutes=30)
    newest_timestamp = datetime.now() - timedelta(minutes=1)

    raw_data = DownloadData(crypto, timeframe=timeframe, interval=[oldest_timestamp, newest_timestamp], limit=100)
    dataframe = pd.DataFrame(raw_data, columns=['DATETIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'timestamp_end', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    dataframe = dataframe[['DATETIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']]

    dataframe['DATETIME'] = pd.to_datetime(dataframe['DATETIME'], unit='ms', utc=True)
    dataframe['DATETIME'] = dataframe['DATETIME'].dt.tz_convert('Europe/Paris')
    dataframe['OPEN'] = dataframe['OPEN'].astype(float)
    dataframe['HIGH'] = dataframe['HIGH'].astype(float)
    dataframe['LOW'] = dataframe['LOW'].astype(float)
    dataframe['CLOSE'] = dataframe['CLOSE'].astype(float)
    dataframe['VOLUME'] = dataframe['VOLUME'].astype(float)

    dataframe = quick_preprocess.fit_transform(dataframe)
    dataframe = dataframe.sort_values(by='DATETIME').reset_index(drop=True)
    return dataframe


def UpdateDataset(crypto, dataset, preprocess):
    timeframe = '1m'
    latest_entry = dataset['DATETIME'].iloc[-1]
    latest_entry = latest_entry.to_pydatetime()
    current_timestamp = datetime.now()

    raw_data = DownloadData(crypto, timeframe=timeframe, interval=[latest_entry, current_timestamp], limit=100)
    update_df = pd.DataFrame(raw_data, columns=['DATETIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'timestamp_end', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    update_df = update_df[['DATETIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']]

    update_df['DATETIME'] = pd.to_datetime(update_df['DATETIME'], unit='ms', utc=True)
    update_df['DATETIME'] = update_df['DATETIME'].dt.tz_convert('Europe/Paris')

    update_df['OPEN'] = update_df['OPEN'].astype(float)
    update_df['HIGH'] = update_df['HIGH'].astype(float)
    update_df['LOW'] = update_df['LOW'].astype(float)
    update_df['CLOSE'] = update_df['CLOSE'].astype(float)
    update_df['VOLUME'] = update_df['VOLUME'].astype(float)

    #ICI
    missing_columns = set(dataset.columns) - set(update_df.columns)
    for col in missing_columns:
        update_df[col] = pd.NA

    dataset = dataset.drop(dataset.tail(1).index)
    combined_df = pd.concat([dataset, update_df]).drop_duplicates(subset=['DATETIME'], keep='last')
    combined_df = combined_df.sort_values(by='DATETIME').reset_index(drop=True)
    combined_df = preprocess.fit_transform(combined_df)
    return combined_df



if __name__ == '__main__':
    try:
        args = Parsing()
#        agents = LoadAgents()
        datasets = {}

        preprocess = Pipeline([
            ('RSI14', RSI(14)),
            ('BLG', BLG(20, 2, live=True)),
        ])   ### FAIRE LE SCALING JUSTE AVANT LA PREDICTION

        for crypto in ActiveCryptos():
            datasets[crypto] = InitDataset(crypto)
        ongoing_timestamp = datetime.now()

        breakpoint()
        while True:
            if ongoing_timestamp.minute != datetime.now().minute:
                for crypto in ActiveCryptos():
                    datasets[crypto] = UpdateDataset(crypto, datasets[crypto], preprocess)
                    X = datasets[crypto][datasets[crypto]['DATETIME'].dt.minute == ongoing_timestamp.minute]
                    breakpoint()
#                    prediction = agent[crypto].predict(X)
#                    if prediction == 0:
#                        take_profit = TakeProfit(X, args.tp)
#                        stop_loss = StopLoss(X, args.sl)
#
#                        Market_trade(crypto, args.risk, stop_loss)  #to do, mettre une protec si il y a plus de capital dispo
#                        OCO_trade(crypto, args.risk, take_profit, stop_loss)#to do
#                last_entry = datetime.now()

    except Exception as error:
        print(error)