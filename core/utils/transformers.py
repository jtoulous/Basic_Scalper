import pandas as pd
import numpy as np
import joblib
import logging
import pickle

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler


class   Cleaner(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_keep):
        self.features_to_keep = features_to_keep

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X[self.features_to_keep]
        X = X.drop(X.index[:10])
        X = X.drop(X.index[-10:])
        X.reset_index(drop=True, inplace=True)
        X.bfill(inplace=True)
        return X


#class   Scaler(BaseEstimator, TransformerMixin):
#    def __init__(self, columns, action='init', path=None):
#        self.columns = columns
#        self.action = action
#        self.path = path
#        self.inited = 0 if action == 'init' else 1
#        self.scaler = StandardScaler() if action == 'init' else joblib.load(path)
#
#    def fit(self, X, y=None):
#        return self
#
#    def transform(self, X, y=None):
#        logging.info("Scaling...")
#
#        X_scaled = X[self.columns]
#        
#        if self.inited == 0 and self.action == 'init':
#            X_scaled = self.scaler.fit_transform(X_scaled)
#            self.inited = 1
#        else:
#            X_scaled = self.scaler.transform(X_scaled)
#        
#        X[self.columns] = X_scaled
#        
#        if self.action == 'init' and self.path is not None:
#            joblib.dump(self.scaler, self.path)
#
#        logging.info("Scaling successful")
#        return X
#


class   Scaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns, load_file=None):
        self.columns = columns
        if load_file is None:
            self.scaler = StandardScaler()
        else:
            with open(load_file, 'rb') as file:
                self.scaler = pickle.load(file)
        self.fitted = 0 if load_file is None else 1

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        logging.info("Scaling...")
        X_scaled = X[self.columns]
        
        if self.fitted == 0:
            breakpoint()
            X_scaled = self.scaler.fit_transform(X_scaled)
            self.fitted = 1
        else:
            breakpoint()
            X_scaled = self.scaler.transform(X_scaled)
        
        X[self.columns] = X_scaled
        
        logging.info("Scaling successful")
        return X



class   RowSelector(BaseEstimator, TransformerMixin):
    def __init__(self, rows):
        self.rows = rows

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X[self.rows]
        return X


class   BalancedOverSampler(BaseEstimator, TransformerMixin):
    def __init__(self, classes):
        self.classes = classes

    def fit(self, X, y=None):
        self.class_counts = {cls: (X['LABEL'] == cls).sum() for cls in self.classes}
        self.max_count = max(self.class_counts.values())
        return self
    
    def transform(self, X, y=None):
        logging.info("Over sampling...")
        for key, value in self.class_counts.items():
            if value < self.max_count:
                target_tmp_df = X[X['LABEL'] == key]
                nb_duplicatas = self.max_count - value
                duplicatas_df = target_tmp_df.sample(n=nb_duplicatas, replace=True).reset_index(drop=True)
                X = pd.concat([X, duplicatas_df], ignore_index=True)
        X = X.sample(frac=1).reset_index(drop=True)
        logging.info("Over sampling successful")
        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


class UnbalancedSampler(BaseEstimator, TransformerMixin):
    def __init__(self, classes, target, minority, percentage):
        self.classes = classes
        self.target = target
        self.minority = minority
        self.unbalance_percentage = percentage

    def fit(self, X, y=None):
        counts = X['LABEL'].value_counts()
        self.class_counts = {}
        for label in self.classes:
            self.class_counts[label] = counts.get(label, 0)
        self.target_objectif = int(self.class_counts[self.minority] * self.unbalance_percentage)
        return self

    def transform(self, X, y=None):
        logging.info(f'Unbalanced over sampling {self.target}...')
#        breakpoint()
        if self.class_counts[self.target] < self.target_objectif:
            target_tmp_df = X[X['LABEL'] == self.target]
            nb_duplicatas = self.target_objectif - self.class_counts[self.target]
            duplicatas_df = target_tmp_df.sample(n=nb_duplicatas, replace=True).reset_index(drop=True)
            X = pd.concat([X, duplicatas_df], ignore_index=True)
        X = X.sample(frac=1).reset_index(drop=True)
        logging.info(f'Unbalanced over sampling {self.target} successful')
        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


class BinSampler(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins=10):
        self.n_bins = n_bins

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        percentiles = np.percentile(X['SCORE'], np.linspace(0, 100, self.n_bins + 1))
        X['BIN'] = pd.cut(X['SCORE'], bins=percentiles, labels=False, include_lowest=True)

        bin_sizes = X['BIN'].value_counts()
        max_bin_size = bin_sizes.max()

        balanced_data = []
        for bin_id in range(self.n_bins):
            bin_data = X[X['BIN'] == bin_id]
            if bin_data.empty:
                continue
            
            current_size = len(bin_data)
            additional_samples = max_bin_size - current_size
            
            if additional_samples > 0:
                oversampled_data = bin_data.sample(n=additional_samples, replace=True, random_state=42)
                bin_data = pd.concat([bin_data, oversampled_data])

            balanced_data.append(bin_data)

        balanced_df = pd.concat(balanced_data, axis=0).drop(columns=['BIN']).reset_index(drop=True)
        balanced_df = balanced_df.sort_values(by='timestamp')
        balanced_df = balanced_df.reset_index(drop=True)
        return balanced_df


class Labeler(BaseEstimator, TransformerMixin):   # 0 for Win, 1 for Lose
    def __init__(self, risk, profit, lifespan):
        self.risk = risk
        self.profit = profit
        self.lifespan = lifespan

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        logging.info("Labelling...")
        X['LABEL'] = 1
#        X['TAKE_PROFIT'] = X['close'] * (1 + self.profit)
#        X['STOP_LOSS'] = X['close'] * (1 - self.risk)
        X = X.sort_values(by='timestamp')

        for idx in range(1, len(X)):
            stop_loss = X['close'].iloc[idx] * (1 - self.risk)
            take_profit = X['close'].iloc[idx] * (1 + self.profit)

            end_idx = min(idx + self.lifespan, len(X))
            for j in range(idx + 1, end_idx):
                open_price = X['open'].iloc[j]
                close_price = X['close'].iloc[j]

                if open_price <= stop_loss:
                    X.loc[idx, 'LABEL'] = 1
                    break

                elif open_price >= take_profit:
                    X.loc[idx, 'LABEL'] = 0
                    break

                elif close_price <= stop_loss:
                    X.loc[idx, 'LABEL'] = 1
                    break

                elif close_price >= take_profit:
                    X.loc[idx, 'LABEL'] = 0
                    break
        logging.info("Labelling successful")
        return X