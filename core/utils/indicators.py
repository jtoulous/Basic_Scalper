import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import hilbert
from .log import printLog


class DateToFeature(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['DAY'] = X['timestamp'].dt.dayofweek + 1
        X['DAY'] = X['DAY'].astype(float)
        return X


class TR(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
 
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        prev_row = None

        for idx, row in X.iterrows():
            if idx == 0:
                X.loc[idx, 'TR'] = row['high'] - row['low']

            else:
                X.loc[idx, 'TR'] = max(row['high'] - row['low'], abs(row['high'] - prev_row['close']), abs(row['low'] - prev_row['close']))
            prev_row = row
        return X


class PriceRange(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
 
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['PR'] = X['high'] - X['low']
        return X


class ATR(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods
 
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        tr = []
        prev_close = None

        for idx, row in X.iterrows():
            if idx == 0:
                tr_val = row['high'] - row['low']
            else:
                tr_val = max(row['high'] - row['low'],
                             abs(row['high'] - prev_close),
                             abs(row['low'] - prev_close))
            tr.append(tr_val)
            prev_close = row['close']

        tr_series = pd.Series(tr, index=X.index)
        X[f'ATR{self.periods}'] = tr_series.rolling(window=self.periods, min_periods=1).mean()
        return X.dropna().reset_index(drop=True)


class ADX(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['H-L_TMP'] = X['high'] - X['low']
        X['H-Close_TMP'] = abs(X['high'] - X['close'].shift(1))
        X['L-Close_TMP'] = abs(X['low'] - X['close'].shift(1))
        X['TR_TMP'] = X[['H-L_TMP', 'H-Close_TMP', 'L-Close_TMP']].max(axis=1)

        delta_high = X['high'].diff(1)
        delta_low = X['low'].diff(1)
        X['+DM_TMP'] = np.where((delta_high > delta_low) & (delta_high > 0), delta_high, 0)
        X['-DM_TMP'] = np.where((delta_low > delta_high) & (delta_low > 0), -delta_low, 0)

        X['+DM_avg_TMP'] = X['+DM_TMP'].rolling(window=self.periods, min_periods=1).mean()
        X['-DM_avg_TMP'] = X['-DM_TMP'].rolling(window=self.periods, min_periods=1).mean()
        X['TR_avg_TMP']    = X['TR_TMP'].rolling(window=self.periods, min_periods=1).mean()

        X['+DI'] = 100 * X['+DM_avg_TMP'] / X['TR_avg_TMP']
        X['-DI'] = 100 * X['-DM_avg_TMP'] / X['TR_avg_TMP']

        DX = 100 * (abs(X['+DI'] - X['-DI']) / (X['+DI'] + X['-DI']))
        X[f'ADX{self.periods}'] = DX.rolling(window=self.periods, min_periods=1).mean()

        X.drop(columns=[
            'H-L_TMP', 'H-Close_TMP', 'L-Close_TMP', 'TR_TMP',
            '+DM_TMP', '-DM_TMP', '+DM_avg_TMP', '-DM_avg_TMP',
            'TR_avg_TMP', '+DI', '-DI'
        ], inplace=True)
        
        return X.dropna().reset_index(drop=True)



class EMA(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):    
        X[f'EMA{str(self.periods)}'] = X['close'].ewm(span=self.periods, adjust=False).mean()
        return X


class EMA_Lagged(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.ema_periods = periods 
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[f'EMA{str(self.ema_periods)}_Lagged'] = X[f'EMA{self.ema_periods}'].shift(1).bfill().ffill()
        return X


class EMA_SMA(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.ema_periods = periods
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[f'EMA{str(self.ema_periods)}_SMA_Ratio'] = X[f'EMA{self.ema_periods}'] / X['SMA'].bfill().ffill()
        return X


class RSI(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        delta = X['close'].diff(1)
        
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=self.periods, min_periods=1).mean()
        avg_loss = loss.rolling(window=self.periods, min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        
        rsi = 100 - (100 / (1 + rs))
        
        rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50)
        rsi = rsi.mask((avg_loss == 0) & (avg_gain != 0), 100)
        
        X[f'RSI{self.periods}'] = rsi
        
        return X.dropna().reset_index(drop=True)


class STO(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        k_periods = self.periods[0]
        d_periods = self.periods[1]

        low_min = X['low'].rolling(window=k_periods, min_periods=1).min()
        high_max = X['high'].rolling(window=k_periods,min_periods=1).max()
        K = ((X['close'] - low_min) / (high_max - low_min)) * 100
        D = K.rolling(window=d_periods, min_periods=1).mean()

        X['K(sto)'] = K.bfill()
        X['D(sto)'] = D.bfill()
        return X
    

class SMA(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[f'SMA{self.periods}'] = np.nan
        X[f'SMA{self.periods}'] = X['close'].rolling(window=self.periods, min_periods=self.periods).mean()
        return X.dropna().reset_index(drop=True)


class SMA_DIFF(BaseEstimator, TransformerMixin):
    def __init__(self, period_1, period_2):
        self.period_1 = period_1
        self.period_2 = period_2

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['tmp_sma_1'] = X['close'].rolling(window=self.period_1, min_periods=1).mean()
        X['tmp_sma_2'] = X['close'].rolling(window=self.period_2, min_periods=1).mean()
        X[f'SMA_DIFF_{self.period_1}_{self.period_2}'] = X['tmp_sma_1'] - X['tmp_sma_2']
        X.drop(columns=['tmp_sma_1', 'tmp_sma_2'], inplace=True)
        return X.dropna().reset_index(drop=True)



class WMA(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        weights = pd.Series(np.arange(1, self.periods + 1) / np.sum(np.arange(1, self.periods + 1)))
        weighted_close = X['close'].rolling(window=self.periods).apply(lambda x: (x * weights).sum(), raw=True)
        X['WMA'] = weighted_close.bfill()
        return X


class DMI(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        prev_row = None
        dm_plus = []
        dm_minus = []

        for idx, row in X.iterrows():
            if prev_row is not None:
                dm_plus_val = row['high'] - prev_row['high'] \
                                if row['high'] - prev_row['high'] > prev_row['low'] - row['low'] \
                                and row['high'] - prev_row['high'] > 0 \
                                else 0
                dm_minus_val = prev_row['low'] - row['low'] \
                                if prev_row['low'] - row['low'] > row['high'] - prev_row['high'] \
                                and prev_row['low'] - row['low'] > 0 \
                                else 0
            else:
                dm_plus_val = 0
                dm_minus_val = 0
            dm_plus.append(dm_plus_val)
            dm_minus.append(dm_minus_val)
            prev_row = row

        dm_plus = pd.Series(dm_plus).bfill()
        dm_minus = pd.Series(dm_minus).bfill()
        dm_diff = abs(dm_plus - dm_minus)
        dm_sum = dm_plus + dm_minus
        dx = dm_diff / dm_sum
        adx = dx.rolling(window=self.periods, min_periods=1).mean().bfill()

        X['DM+'] = dm_plus
        X['DM-'] = dm_minus
        X['ADX'] = adx
        return X


class BLG(BaseEstimator, TransformerMixin):
    def __init__(self, periods, num_std_dev):
        self.periods = periods
        self.num_std_dev = num_std_dev

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        TMP_SMA = X['close'].rolling(window=self.periods, min_periods=1).mean()
        TMP_STD = X['close'].rolling(window=self.periods, min_periods=1).std()
        
        u_band = TMP_SMA + (TMP_STD * self.num_std_dev)
        l_band = TMP_SMA - (TMP_STD * self.num_std_dev)
        
        X['U-BAND'] = u_band
        X['L-BAND'] = l_band
        X['BLG_WIDTH'] = u_band - l_band

        return X.dropna().reset_index(drop=True)


class MACD(BaseEstimator, TransformerMixin):
    def __init__(self, ema_short, ema_long, ema_signal):
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.ema_signal = ema_signal

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        short_period = self.ema_short
        long_period = self.ema_long
        signal_period = self.ema_signal

        ema_short = X['close'].ewm(span=short_period, adjust=False, min_periods=short_period).mean()
        ema_long = X['close'].ewm(span=long_period, adjust=False, min_periods=long_period).mean()
        macd_line = ema_short - ema_long
        macd_signal = macd_line.ewm(span=signal_period, adjust=False, min_periods=signal_period).mean()
        macd_histo = macd_line - macd_signal

        X['MACD_LINE'] = macd_line
        X['MACD_SIGNAL'] = macd_signal
        X['MACD_HISTO'] = macd_histo

        return X.dropna().reset_index(drop=True)


class HilbertsTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):    
        analytic_signal = hilbert(X['close'])
        X['HBT_TRANS'] = analytic_signal.imag
        X['HBT_TRANS'] = X['HBT_TRANS'].bfill()
        return X


class CCI(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        TP = (X['high'] + X['low'] + X['close']) / 3
        SMA_TP = TP.rolling(window=self.periods, min_periods=1).mean()
        MD = TP.rolling(window=self.periods, min_periods=1).apply(lambda x: np.mean(np.abs(x - np.mean(x))))

        CCI = (TP -SMA_TP) / (0.015 * MD)
        X['CCI'] = CCI.bfill()
        return X


class PPO(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        short_period = self.periods[0]
        long_period = self.periods[1]
        signal_period = self.periods[2]

        short_ema = X['close'].ewm(span=short_period, min_periods=1).mean()
        long_ema = X['close'].ewm(span=long_period, min_periods=1).mean()
        ppo_line = ((short_ema - long_ema) / long_ema) * 100
        signal_line = ppo_line.ewm(span=signal_period, min_periods=1).mean()

        X['PPO_LINE'] = ppo_line.bfill().ffill()
        X['PPO_SIGNAL'] = signal_line.bfill().ffill()
        return X


class ROC(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['ROC'] = X['close'].pct_change()
        X['ROC'] = X['ROC'].bfill()
        return X



class Slope(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        slopes = []

        for i in range(self.periods, len(X)):
            y_vals = X['close'].iloc[i - self.periods:i]
            x_vals = range(self.periods)
            slope, intercept = np.polyfit(x_vals, y_vals, 1)
            slopes.append(slope)

        slopes = [np.nan] * (self.periods) + slopes
        X['SLOPE'] = slopes
        return X




class Z_SCORE(BaseEstimator, TransformerMixin):
    def __init__(self, periods, column):
        self.periods = periods
        self.column = column
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        col_mean = X[self.column].rolling(window=self.periods, min_periods=1).mean()
        col_std = X[self.column].rolling(window=self.periods, min_periods=1).std()

        X[f'Z_SCORE_{self.column}'] = (X[self.column] - col_mean) / col_std
        return X.dropna().reset_index(drop=True)


class   Growth(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['GROWTH'] = (X['close'] - X['volume']) / X['volume'] * 100
        return X


class HV(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['log_returns'] = np.log(X['close'] / X['close'].shift(1))
        X[f'HV{self.periods}'] = X['log_returns'].rolling(window=self.periods).std()
        
        X.drop(columns=['log_returns'], inplace=True)
        
        return X


class OBV(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['OBV'] = np.nan
        
        for i in range(1, len(X)):
            if X['close'].iloc[i] > X['close'].iloc[i - 1]:
                X.loc[i, 'OBV'] = X['volume'].iloc[i]
            elif X['close'].iloc[i] < X['close'].iloc[i - 1]:
                X.loc[i, 'OBV'] = -X['volume'].iloc[i]
            else:
                X.loc[i, 'OBV'] = 0

        X['OBV'] = X['OBV'].cumsum()
        X = X.dropna(subset=['OBV']).reset_index(drop=True)
        return X



class CMF(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['CMF'] = np.nan

        for i in range(len(X)):
            if X['high'].iloc[i] != X['low'].iloc[i]:
                money_flow_multiplier = ((X['close'].iloc[i] - X['low'].iloc[i]) - (X['high'].iloc[i] - X['close'].iloc[i])) / (X['high'].iloc[i] - X['low'].iloc[i])
                money_flow_volume = money_flow_multiplier * X['volume'].iloc[i]
                X.loc[i, 'CMF'] = money_flow_volume

        X['CMF'] = X['CMF'].rolling(window=self.periods, min_periods=1).sum() / X['volume'].rolling(window=self.periods, min_periods=1).sum()
        X = X.dropna(subset=['CMF']).reset_index(drop=True)

        return X


class RealizedVolatility(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[f'VOLATILITY{self.periods}'] = np.nan
        X[f'log_returns'] = np.nan
        
        X['log_returns'] = np.log(X['close'] / X['close'].shift(1))
        X[f'VOLATILITY{self.periods}'] = X['log_returns'].rolling(window=self.periods).std() * np.sqrt(self.periods)
        X.drop(columns=['log_returns'], inplace=True)
        
        X = X.dropna(subset=[f'VOLATILITY{self.periods}']).reset_index(drop=True)
        return X



class DailyLogReturn(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['LOG_RTN'] = np.log(X['close'] / X['close'].shift(1))
        X = X.dropna().reset_index(drop=True)
        return X        




class HL_Ratio(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['HL_Ratio'] = (X['high'] - X['low']) / X['close']
        return X.dropna().reset_index(drop=True)



class VolumeRatio(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[f'Volume_Ratio_{self.periods}'] = X['volume'] / X['volume'].rolling(window=self.periods, min_periods=1).mean()
        return X.dropna().reset_index(drop=True)









class SwingLabeler(BaseEstimator, TransformerMixin):   # 0 for Win, 1 for Lose, ATR must be calculated before hand
    def __init__(self, risk=0.5, profit=1, lifespan=10):
        self.risk = risk
        self.profit = profit
        self.lifespan = lifespan

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
#        logging.info("Labelling...")
        X['LABEL'] = 1
        X['TAKE_PROFIT'] = X['close'] + (X['ATR14'] * self.profit)
        X['STOP_LOSS'] = X['close'] - (X['ATR14'] * self.risk)
        X = X.sort_values(by='timestamp')

        for idx in range(len(X) - 1):
            tp = X.loc[X.index[idx], 'TAKE_PROFIT']
            sl = X.loc[X.index[idx], 'STOP_LOSS']
            label_set = False

            for j in range(idx + 1, min(idx + self.lifespan, len(X))):
                row = X.loc[X.index[j]]

                if row['open'] >= tp:
                    X.loc[X.index[idx], 'LABEL'] = 0
                    label_set = True
                    break
                elif row['open'] <= sl:
                    X.loc[X.index[idx], 'LABEL'] = 1
                    label_set = True
                    break

                if row['high'] >= tp:
                    X.loc[X.index[idx], 'LABEL'] = 0
                    label_set = True
                    break

                if row['low'] <= sl:
                    X.loc[X.index[idx], 'LABEL'] = 1
                    label_set = True
                    break

                if row['close'] >= tp:
                    X.loc[X.index[idx], 'LABEL'] = 0
                    label_set = True
                    break

                if row['close'] <= sl:
                    X.loc[X.index[idx], 'LABEL'] = 1
                    label_set = True
                    break

            if not label_set:
                X.loc[X.index[idx], 'LABEL'] = 1

        return X