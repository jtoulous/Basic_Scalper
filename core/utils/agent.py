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
        self.features = ['RSI14', 'U-BAND', 'L-BAND']
        self.crypto = crypto
        self.models = {}
        self.scaler = Scaler(self.features, self.crypto, 'train') if load is False else Scaler(self.features, self.crypto, 'predict')

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


    def train(self, dataframe, crossval=False):#Fine tuner le modele sur les 3-4 dernieres annees a la fin
            preprocess = Pipeline([
                ('RSI', RSI(14)),
                ('BLG', BLG(20, 2)),
                ('Labeler', Labeler(risk=0.001, profit=0.002, lifespan=10)),
                ('Scaler', self.scaler),
                ('Sampler', BalancedOverSampler([0, 1])),
            ])
            logging.info(f'Starting {self.crypto} preprocessing')
            dataframe = preprocess.fit_transform(dataframe)
            dataframe = dataframe.sort_values(by='DATETIME')
            dataframe = dataframe.reset_index(drop=True)
            logging.info("Preprocessing completed")

            X, y = dataframe[self.features], dataframe['LABEL']
            if crossval is True:
                CrossVal(self.models['MLP_Balanced'], X, y)

            logging.info(f'Training...')
            self.models['MLP_Balanced'].fit(X, y)
            logging.info(f'Training done')



    def predict(self, dataframe):
            preprocess = Pipeline([
                ('RSI', RSI(14)),
                ('BLG', BLG(20, 2)),
                ('Scaler', self.scaler),
            ])
            dataframe = preprocess.transform(dataframe)
            X = dataframe[self.features]

            prediction = self.models['MLP_Balanced'].predict(X)
            return prediction


    def save(self):
        joblib.dump(f'data/agents/{self.crypto}_agent.pkl')