import pandas as pd
import xgboost as xgb
import joblib
import logging
import pickle
from colorama import Fore, Style

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from utils.indicators import RSI, BLG
from utils.transformers import Labeler, Scaler, BalancedOverSampler, UnbalancedSampler
from utils.tools import CrossVal



class Agent():
    def __init__(self, features, load_file=None, scaler_file=None, crossval=False):
        self.features = features
        self.crossval = crossval
        self.models = {}

        if load_file is None:
#            self.scaler = Scaler(self.features)
            self.scaler = StandardScaler()


#            self.models['MLP_Balanced'] = LogisticRegression(
#                solver='liblinear', 
#                random_state=42,
#                max_iter=500
#            )

            self.models['MLP_Balanced'] = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                learning_rate='adaptive',
                max_iter=1000,
                early_stopping=True,
                random_state=42,
                batch_size=32
            )

#            self.models['MLP_Unbalanced_0'] = MLPClassifier(
#                hidden_layer_sizes=(128, 64, 32),
#                activation='relu',
#                solver='adam',
#                alpha=0.0001,
#                learning_rate='adaptive',
#                max_iter=1000,
#                early_stopping=True,
#                random_state=42,
#                batch_size=32
#            )
#
#            self.models['MLP_Unbalanced_1'] = MLPClassifier(
#                hidden_layer_sizes=(128, 64, 32),
#                activation='relu',
#                solver='adam',
#                alpha=0.0001,
#                learning_rate='adaptive',
#                max_iter=1000,
#                early_stopping=True,
#                random_state=42,
#                batch_size=32
#            )
#
#            self.models['MLP_Master'] = MLPClassifier(
#                hidden_layer_sizes=(128, 64, 32),
#                activation='relu',
#                solver='adam',
#                alpha=0.0001,
#                learning_rate='adaptive',
#                max_iter=1000,
#                early_stopping=True,
#                random_state=42,
#                batch_size=32
#            )

        else:
            with open(load_file, 'rb') as file:
                self.models = pickle.load(file)
            with open(scaler_file, 'rb') as file:
                self.scaler = pickle.load(file)


    def train(self, dataframe):#Fine tuner le modele sur les 3-4 dernieres annees a la fin
        dataframe = dataframe.copy()
        
        logging.info(f'Scaling data')
        dataframe[self.features] = self.scaler.fit_transform(dataframe[self.features])
        logging.info("Scaling completed")

        self.train_balanced(dataframe.copy())
#        self.train_unbalanced(dataframe.copy(), 0)
#        self.train_unbalanced(dataframe.copy(), 1)

#        df_master = self.combine_predictions(dataframe[self.features])
#        df_master['LABEL'] = dataframe['LABEL']
#        df_master = df_master.sample(frac=1).reset_index(drop=True)
        
#        X_master = df_master[['PRED_1', 'PRED_2', 'PRED_3']]
#        y_master = dataframe['LABEL']
#        if self.crossval is True:
#            logging.info(f'crossval Master...')
#            CrossVal(self.models['MLP_Master'], X_master, y_master)

#        logging.info(f'Training Master...')
#        self.models['MLP_Master'].fit(X_master, y_master)
#        logging.info(f'Training successful')


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
            dataframe = dataframe.copy()
            dataframe[self.features] = self.scaler.transform(dataframe[self.features])
#            X_master = self.combine_predictions(dataframe[self.features])

#            prediction = self.models['MLP_Master'].predict(X_master)
            prediction = self.models['MLP_Balanced'].predict(dataframe[self.features])
            return prediction


    def predict_validate(self, dataframe):
        dataframe = dataframe.copy()
        dataframe[self.features] = self.scaler.transform(dataframe[self.features])
#        X_master = self.combine_predictions(dataframe[self.features])
#        predictions = self.models['MLP_Master'].predict(X_master)
        predictions = self.models['MLP_Balanced'].predict(dataframe[self.features])

        nb_correct_preds = 0
        nb_uncorrect_preds = 0
        nb_win_preds = 0
        nb_true_win = 0
        for idx, row in dataframe[['LABEL']].iterrows():
            pred = predictions[idx]
            if pred == row['LABEL']:
                nb_correct_preds = nb_correct_preds + 1
                print(f'{Fore.GREEN}{row['LABEL']}  ==>  {pred}{Style.RESET_ALL}')
            else:
                print(f'{Fore.RED}{row['LABEL']}  ==>  {pred}{Style.RESET_ALL}')
            
            if pred == 0:
                nb_win_preds = nb_win_preds + 1
                if row['LABEL'] == 0:
                    nb_true_win = nb_true_win + 1
        
        if nb_win_preds > 0:
            print(f'Validation result:\n - {(nb_correct_preds / len(dataframe) * 100):.2f}% correct predictions\n - {(nb_true_win / nb_win_preds * 100):.2f}%({nb_true_win}/{nb_win_preds}) true wins')
        else:
            print(f'Validation result:\n  - {(nb_correct_preds / len(dataframe)) * 100}% correct predictions')


    def combine_predictions(self, X):
        X_master = pd.DataFrame()
        X_master['PRED_1'] = self.models['MLP_Balanced'].predict(X)
        X_master['PRED_2'] = self.models['MLP_Unbalanced_0'].predict(X)
        X_master['PRED_3'] = self.models['MLP_Unbalanced_1'].predict(X)
        return X_master


    def save(self, models_path, scaler_path):
        with open(models_path, 'wb') as file:
            pickle.dump(self.models, file)

        with open(scaler_path, 'wb') as file:
            pickle.dump(self.scaler, file)