import logging
import pandas as pd
import numpy as np

from collections import Counter
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score

from utils.log import printLog


def CrossVal(model, X, y, cv=5):
    printLog(' ===> Cross validation...')
    scores = cross_val_score(model, X, y, cv=cv)
    printLog(f'   ==> Cross-Validation Scores: {scores}')
    printLog(f'   ==> Average Accuracy: {scores.mean()}')



def TimeSeriesSplitValidation(X, y, models):
        tscv = TimeSeriesSplit(n_splits=5)
        
        mlp_mse_scores, gbr_mse_scores, rfr_mse_scores, svr_mse_scores, xgb_mse_scores, ext_mse_scores = [], [], [], [], [], []
        mlp_r2_scores, gbr_r2_scores, rfr_r2_scores, svr_r2_scores, xgb_r2_scores, ext_r2_scores = [], [], [], [], [], []

        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            models['XGB'].fit(X_train, y_train)

            xgb_pred = models['XGB'].predict(X_test)
            xgb_mse_scores.append(mean_squared_error(y_test, xgb_pred))
            xgb_r2_scores.append(r2_score(y_test, xgb_pred))

        xgb_mse = np.mean(xgb_mse_scores)
        xgb_r2 = np.mean(xgb_r2_scores)
        logging.info(f"XGB - Cross-validation MSE: {xgb_mse}, R²: {xgb_r2}")


def TakeProfit(X, profit):
    take_profit = X['CLOSE'] * (1 + profit)
    return take_profit


def StopLoss(X, risk):
    stop_loss = X['CLOSE'] * (1 - risk)
    return stop_loss