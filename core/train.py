import argparse
import logging
import pandas as pd

from utils.agent import Agent
from utils.dataframe import ReadDf
from utils.arguments import ActiveCryptos

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def Parsing():
    parser = argparse.ArgumentParser(description="Options for running the trading script")
    parser.add_argument('-crossval', action='store_true', help='Apply cross-validation')
    return parser.parse_args()

if __name__ == '__main__':
    try:
        args = Parsing()

        for crypto in ActiveCryptos():
            agent = Agent(crypto)
#            dataframe = ReadDf(f'data/crypto/{crypto}/1m/{crypto}_1m.csv')
#            dataframe = ReadDf(f'data/crypto/{crypto}/1m/{crypto}_23-24.csv')
            dataframe = pd.read_csv(f'data/crypto/{crypto}/1m/{crypto}_preprocessed.csv')
            
#            dataframe = dataframe[(dataframe['DATETIME'] >= '2020-01-01') & (dataframe['DATETIME'] <= '2024-12-10')]
#            dataframe = dataframe.reset_index(drop=True)

            logging.info(f'Training {crypto} agent')
            agent.train(dataframe.copy(), args.crossval)
            logging.info(f'Agent {crypto} trained successfully')

            logging.info("Saving agent")
            agent.save(X, y)
            logging.info(f'Agent {crypto} saved successfully')

    except Exception as error:
        logging.error("Error: %s", error, exc_info=True)
