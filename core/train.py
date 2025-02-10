import argparse
import logging
import pandas as pd

from utils.agent import Agent
from utils.dataframe import ReadDf
from utils.arguments import ActiveCryptos
from utils.transformers import Labeler
from utils.indicators import RSI, BLG, ATR, ADX, DailyLogReturn, SMA, SMA_DIFF, MACD, Z_SCORE, HL_Ratio, VolumeRatio, SwingLabeler


from sklearn.pipeline import Pipeline

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
            agent = Agent(features=[
                'LOG_RTN',
                'SMA5',
                'SMA20',
                'SMA_DIFF_5_20',
                'MACD_LINE',
                'MACD_SIGNAL',
                'BLG_WIDTH',
                'ATR14',
                'RSI14',
                'ADX14',
                'Z_SCORE_close',
                'HL_Ratio',
                'Volume_Ratio_20'
            ], crossval=args.crossval)
            
            dataframe = pd.read_csv(f'data/crypto/BTCUSDT/1d/BTCUSDT_1d.csv')
            preprocess = Pipeline([
                ('Daily log rtn', DailyLogReturn()),#
                ('SMA 5', SMA(5)),#
                ('SMA 20', SMA(20)),#
                ('SMA DIFF', SMA_DIFF(5, 20)),#
                ('MACD', MACD(12, 26, 9)),#
                ('BLG WIDTH', BLG(20, 20)),
                ('ATR 14', ATR(14)),
                ('RSI 14', RSI(14)),
                ('ADX 14', ADX(14)),
                ('Z_score', Z_SCORE(20, 'close')),
                ('Ratio H_L', HL_Ratio()),
                ('Volume ratio', VolumeRatio(20)),
                ('Labeler', SwingLabeler(risk=0.5, profit=1, lifespan=1)),
            ])
            dataframe = preprocess.fit_transform(dataframe)
            breakpoint()
            dataframe = dataframe[26:]
            df_train = dataframe[:int(len(dataframe) * 0.8)]
            df_test = dataframe[int(len(dataframe) * 0.8):].reset_index(drop=True)

            df_train.to_csv('df_train.csv', index=False)
            df_test.to_csv('df_test.csv', index=False)

            logging.info(f'Training {crypto} agent')
            agent.train(df_train)
            logging.info(f'Agent {crypto} trained successfully')

            logging.info("Saving agent")
            agent.save(f'data/agents/{crypto}_models.pkl', f'data/agents/scalers/{crypto}_scaler.pkl')
            logging.info(f'Agent {crypto} saved successfully')

            agent.predict_validate(df_test)

    except Exception as error:
        logging.error("Error: %s", error, exc_info=True)
