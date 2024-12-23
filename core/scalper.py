import logging
import argparse
import pandas as pd
from datetime import datetime

from utils.agent import Agent
from utils.arguments import ActiveCryptos
from utils.tools import TakeProfit, StopLoss
from utils.binance import DownloadData

def Parsing():
    parser = ArgumentParser()
    parser.add_argument('-risk', type=float, default=0.01, help='percentage of capital at risk on each trades')
    parser.add_argument('-sl', type=float, default=0.001, help='to calculate stop loss, use the same as training')
    parser.add_argument('-tp', type=float, default=0.002, help='to calculate take profit, use the same as training')
    args = parser.parse_args()
    return args



def LoadAgents():
    agents = {}
    for crypto in ActiveCryptos:
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


if __name__ == '__main__':
    try:
        args = Parsing()
        agents = LoadAgents()

        last_datetime = datetime.now()
        while True:
            if last_datetime.minute != datetime.now().minute:
                for crypto in ActiveCryptos():
                    X = DownloadData(crypto, last_datetime)
                    take_profit = TakeProfit(X, args.tp)
                    stop_loss = StopLoss(X, args.sl)

                    prediction = agent[crypto].predict(X)
                    if prediction == 0: # 0 = Win 
                        Market_trade(crypto, args.risk, stop_loss)  #to do, mettre une protec si il y a plus de capital dispo
                        OCO_trade(crypto, args.risk, take_profit, stop_loss)#to do
                
                last_datetime = datetime.now()

    except Exception as error:
        print(error)