import pandas as pd
import xgboost as xgb
import joblib
import logging

from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

from utils.indicators import RSI, BLG
from utils.transformers import Labeler, Scaler, BalancedOverSampler, UnbalancedSampler
from utils.tools import CrossVal

from sklearn.ensemble import RandomForestClassifier


class Agent():
    def __init__(self, crypto, load=False, crossval=False):
        self.features = ['RSI14', 'U-BAND', 'L-BAND']
        self.crypto = crypto
        self.crossval = crossval
        self.models = {}
        self.scaler = Scaler(self.features, action='init', path=f'data/agents/scalers/{crypto}_scaler.pkl') if load is False else Scaler(self.features, action='load', path=f'data/agents/scalers/{crypto}_scaler.pkl')

        if load is False:
            self.models['MLP_Balanced'] = MLPClassifier(
                hidden_layer_sizes=(32, 16),
                activation='relu',
                solver='adam',
                random_state=42,
                max_iter=500
            )

            self.models['MLP_Unbalanced_0'] = MLPClassifier(
                hidden_layer_sizes=(32, 16),
                activation='relu',
                solver='adam',
                random_state=42,
                max_iter=500
            )

            self.models['MLP_Unbalanced_1'] = MLPClassifier(
                hidden_layer_sizes=(32, 16),
                activation='relu',
                solver='adam',
                random_state=42,
                max_iter=500
            )

            self.models['MLP_Master'] = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                solver='adam',
                random_state=42,
                max_iter=1000
            )

        else:
            self.models['MLP_Balanced'] = joblib.load(f'data/agents/{crypto}_balanced.pkl')
            self.models['MLP_Unbalanced_0'] = joblib.load(f'data/agents/{crypto}_unbalanced_0.pkl')
            self.models['MLP_Unbalanced_1'] = joblib.load(f'data/agents/{crypto}_unbalanced_1.pkl')
            self.models['MLP_Master'] = joblib.load(f'data/agents/{crypto}_master.pkl')


    def train(self, dataframe):#Fine tuner le modele sur les 3-4 dernieres annees a la fin
        preprocess = Pipeline([
            ('RSI', RSI(14)),
            ('BLG', BLG(20, 2)),
            ('Labeler', Labeler(risk=0.001, profit=0.002, lifespan=10)),
            ('Scaler', self.scaler),
        ])

        logging.info(f'Starting {self.crypto} preprocessing')
        dataframe = preprocess.fit_transform(dataframe)
        logging.info("Preprocessing completed")

        self.train_balanced(dataframe.copy())
        self.train_unbalanced(dataframe.copy(), 0)
        self.train_unbalanced(dataframe.copy(), 1)

        df_master = self.combine_predictions(dataframe[self.features])
        df_master['LABEL'] = dataframe['LABEL']
        df_master = df_master.sample(frac=1).reset_index(drop=True)
        
        X_master = df_master[['PRED_1', 'PRED_2', 'PRED_3']]
        y_master = dataframe['LABEL']
        if self.crossval is True:
            logging.info(f'crossval Master...')
            CrossVal(self.models['MLP_Master'], X_master, y_master)

        logging.info(f'Training Master...')
        self.models['MLP_Master'].fit(X_master, y_master)
        logging.info(f'Training successful')


    def train_balanced(self, dataframe):
        preprocess_pipeline = Pipeline([
            ('sampler', BalancedOverSampler([0, 1])),
        ])
        df = preprocess_pipeline.fit_transform(dataframe)
        X, y = df[self.features], df['LABEL']

        if self.crossval is True:
            logging.info(f'crossval balanced...')
            CrossVal(self.models['MLP_Balanced'], X, y)
        self.models['MLP_Balanced'].fit(X, y)


    def train_unbalanced(self, dataframe, b_type):
        majority = 0 if b_type == 0 else 1
        minority = 0 if majority == 1 else 1
        preprocess_pipeline = Pipeline([
            ('sampler', UnbalancedSampler([0, 1], majority, minority, 1.5))
        ])
        df = preprocess_pipeline.fit_transform(dataframe)
        X, y = df[self.features], df['LABEL']

        if self.crossval is True:
            logging.info(f'crossval unbalanced {b_type}...')
            CrossVal(self.models[f'MLP_Unbalanced_{b_type}'], X, y)
        self.models[f'MLP_Unbalanced_{b_type}'].fit(X, y)



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




    def predict_validate(self, dataframe):
        preprocess = Pipeline([
            ('RSI', RSI(14)),
            ('BLG', BLG(20, 2)),
            ('Scaler', self.scaler),
        ])
        dataframe




    def combine_predictions(self, X):
        X_master = pd.DataFrame()
        X_master['PRED_1'] = self.models['MLP_Balanced'].predict(X)
        X_master['PRED_2'] = self.models['MLP_Unbalanced_0'].predict(X)
        X_master['PRED_3'] = self.models['MLP_Unbalanced_1'].predict(X)
        return X_master


    def save(self):
        joblib.dump(self.models['MLP_Balanced'], f'data/agents/{self.crypto}_balanced.pkl')
        joblib.dump(self.models['MLP_Unbalanced_0'], f'data/agents/{self.crypto}_unbalanced_0.pkl')
        joblib.dump(self.models['MLP_Unbalanced_1'], f'data/agents/{self.crypto}_unbalanced_1.pkl')
        joblib.dump(self.models['MLP_Master'], f'data/agents/{self.crypto}_master.pkl')