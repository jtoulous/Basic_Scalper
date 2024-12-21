import pandas as pd
import xgboost as xgb
import joblib
import logging

from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

from utils.indicators import RSI, BLG
from utils.transformers import Labeler, Scaler, BalancedOverSampler
from utils.tools import CrossVal


class Agent():
    def __init__(self, crypto, load=False):
        self.crypto = crypto
        self.models = {}

        if load is False:
            self.models['MLP_Balanced'] = MLPClassifier(      #Model trained on balanced data
                hidden_layer_sizes=(64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size=32,
                learning_rate='adaptive',
                max_iter=200,
                random_state=42
            )



        else:
            self.models['MLP_Balanced'] = joblib.load(f'data/agents/{crypto}_agent.pkl')


    def train(self, dataframe, crypto, crossval=False):#Fine tuner le modele sur les 3-4 dernieres annees a la fin
            features = ['RSI14', 'U-BAND', 'L-BAND']
            preprocess = Pipeline([
                ('RSI', RSI(14)),
                ('BLG', BLG(20, 2)),
                ('Labeler', Labeler(risk=0.001, profit=0.002, lifespan=10)),
                ('Scaler', Scaler(features, crypto, 'train')),
                ('Sampler', BalancedOverSampler([0, 1])),
            ])
            logging.info(f'Starting {crypto} preprocessing')
            dataframe = preprocess.fit_transform(dataframe)
            dataframe = dataframe.sort_values(by='DATETIME')
            dataframe = dataframe.reset_index(drop=True)
            logging.info("Preprocessing completed")

            breakpoint()

            X, y = dataframe[features], dataframe['LABEL']
            if crossval is True:
                CrossVal(self.models['MLP_Balanced'], X, y)

            breakpoint()
            logging.info(f'Training...')
            self.models['MLP_Balanced'].fit(X, y)
            logging.info(f'Training done')



#    def predict(self, X)

    def save(self):
        joblib.dump(f'data/agents/{self.crypto}_agent.pkl')