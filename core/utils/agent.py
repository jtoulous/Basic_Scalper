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
    def __init__(self, crypto, load=False):
        self.features = ['RSI14', 'U-BAND', 'L-BAND']
        self.crypto = crypto
        self.models = {}
        self.scaler = Scaler(self.features, self.crypto, 'train') if load is False else Scaler(self.features, self.crypto, 'predict')

        if load is False:
            self.models['RF_Balanced'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        
            self.models['RF_Unbalanced_0'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )

            self.models['RF_Unbalanced_1'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )

            self.models['RF_Master'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )

        else:
            self.models['RF_Balanced'] = joblib.load(f'data/agents/{crypto}_balanced.pkl')
            self.models['RF_Unbalanced_0'] = joblib.load(f'data/agents/{crypto}_unbalanced_0.pkl')
            self.models['RF_Unbalanced_1'] = joblib.load(f'data/agents/{crypto}_unbalanced_1.pkl')
            self.models['RF_Master'] = joblib.load(f'data/agents/{crypto}_master.pkl')


    def train(self, dataframe, crossval=False):#Fine tuner le modele sur les 3-4 dernieres annees a la fin
#            preprocess = Pipeline([
#                ('RSI', RSI(14)),
#                ('BLG', BLG(20, 2)),
#                ('Labeler', Labeler(risk=0.001, profit=0.002, lifespan=10)),
#                ('Scaler', self.scaler),
##                ('Gan', GAN())
#            ])
            breakpoint()   
            balanced_oversampler = BalancedOverSampler([0, 1])
            unbalanced_oversampler_0 = UnbalancedSampler([0, 1], 0, 1, 1.5)
            unbalanced_oversampler_1 = UnbalancedSampler([0, 1], 1, 0, 1.5)

#            logging.info(f'Starting {self.crypto} preprocessing')
#            dataframe = preprocess.fit_transform(dataframe)
#            logging.info("Preprocessing completed")
            breakpoint()
            
            dataframe_balanced = balanced_oversampler.fit_transform(dataframe.copy())
            dataframe_unbalanced_0 = unbalanced_oversampler_0.fit_transform(dataframe.copy())   # A VERIFIER
            dataframe_unbalanced_1 = unbalanced_oversampler_1.fit_transform(dataframe.copy())   # A VERIFIER
            breakpoint()   ####     CHECK VALUES OF SAMPLING, might be good, but weird looking atm

            X_balanced, y_balanced = dataframe_balanced[self.features], dataframe_balanced['LABEL']
            X_unbalanced_0, y_unbalanced_0 = dataframe_unbalanced_0[self.features], dataframe_unbalanced['LABEL']
            X_unbalanced_1, y_unbalanced_1 = dataframe_unbalanced_1[self.features], dataframe_unbalanced['LABEL']
            breakpoint()
            
            if crossval is True:
                CrossVal(self.models['RF_Balanced'], X_balanced, y_balanced)
                CrossVal(self.models['RF_Unbalanced_0'], X_unbalanced_0, y_unbalanced_0)
                CrossVal(self.models['RF_Unbalanced_1'], X_unbalanced_1, y_unbalanced_1)
            breakpoint()

            logging.info(f'Training sub-models...')
            self.models['RF_Balanced'].fit(X_balanced, y_balanced)
            self.models['RF_Unbalanced_0'].fit(X_unbalanced_0, y_unbalanced_0)
            self.models['RF_Unbalanced_1'].fit(X_unbalanced_1, y_unbalanced_1)
            logging.info(f'Training successful')

            X_master = self.combine_predictions(X_balanced)
            y_master = y_balanced
            if crossval is True:
                CrossVal(self.models['RF_Master'], X_master, y_master)

            logging.info(f'Training Master...')
            self.models['RF_Master'].fit(X_master, y_master)
            logging.info(f'Training successful')


#    def predict(self, dataframe):
#            preprocess = Pipeline([
#                ('RSI', RSI(14)),
#                ('BLG', BLG(20, 2)),
#                ('Scaler', self.scaler),
#            ])
#            dataframe = preprocess.transform(dataframe)
#            X = dataframe[self.features]
#
#            prediction = self.models['RF_Balanced'].predict(X)
#            return prediction


    def combine_predictions(self, X):
        X_master = pd.DataFrame()
        X_master['PRED_1'] = self.models['RF_Balanced'].predict(X)
        X_master['PRED_2'] = self.models['RF_Unbalanced_0'].predict(X)
        X_master['PRED_3'] = self.models['RF_Unbalanced_1'].predict(X)
        return X_master



    def save(self):
        joblib.dump(f'data/agents/{self.crypto}_balanced.pkl')
        joblib.dump(f'data/agents/{self.crypto}_unbalanced_1.pkl')
        joblib.dump(f'data/agents/{self.crypto}_unbalanced_2.pkl')
        joblib.dump(f'data/agents/{self.crypto}_master.pkl')
