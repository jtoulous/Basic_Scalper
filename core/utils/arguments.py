import pandas as pd


def ActiveCryptos():
    active_cryptos = [
        'BTCUSDT',
#        'ETHUSDT',
#        'SOLUSDT',
#        'BNBUSDT',
#        'ADAUSDT',
#        'DOGEUSDT',
#        'XRPUSDT',
#        'LTCUSDT',
    ]
    return active_cryptos

#def GetOpen(args, dataframe, crypto):
#    if args.date is not None:
#        date = pd.to_datetime(args.date, format='%d/%m/%Y')
#    else:
#        date = dataframe.iloc[-1]['DATETIME'] + pd.DateOffset(days=1)
#    data = yf.download(crypto, start=date, interval='1d')
#    if len(data) < 1:    
#        return dataframe.iloc[-1]['CLOSE']
#    return data.iloc[0]['Open']


def GetDate(args, dataframe):
    if args.date is not None:
        return pd.to_datetime(args.date, format='%d/%m/%Y')
    return dataframe.iloc[-1]['DATETIME'] + pd.DateOffset(days=1)


def GetCryptoFile(crypto, interval, file_type='default', path=None):
    if file_type == 'default':
        return f'data/CRYPTOS/{crypto}/{interval}/{crypto}_{interval}.csv'
#        return f'data/CRYPTOS/{crypto}/{interval}/train.csv'

    if file_type == 'test train':
        if interval == '1d':
            return f'data/CRYPTOS/{crypto}/{interval}/test_train.csv'
        elif interval == '30m':
            return f'data/CRYPTOS/{crypto}/{interval}/test_train.csv'

    if file_type == 'test predict':
        if interval == '1d':
            return f'data/CRYPTOS/{crypto}/{interval}/test_predict.csv'
        elif interval == '30m':
            return f'data/CRYPTOS/{crypto}/{interval}/test_predict.csv'

    if file_type == 'visualize':
        return f'data/CRYPTOS/{crypto}/{interval}/{crypto}_{interval}.csv'

#def GetArg(arg_type):  ## params for 1/3 R/P
#    if arg_type == 'lifespan':
#        return 50
#    elif arg_type == 'atr':
#        return 50
#    elif arg_type == 'ema':
#        return 200
#    elif arg_type == 'rsi':
#        return 30
#    elif arg_type == 'sto':
#        return [30, 10]
#    elif arg_type == 'sma':
#        return 200
#    elif arg_type == 'wma':
#        return 50
#    elif arg_type == 'dmi':
#        return 30
#    elif arg_type == 'blg':
#        return [50, 3]
#    elif arg_type == 'macd':
#        return [50, 100, 20]
#    elif arg_type == 'cci':
#        return 50
#    elif arg_type == 'ppo':
#        return [50, 100, 20]


#def GetRP(crypto, arg_type):
#    if crypto == 'BTC-USD':
#        return 1 if arg_type == 'R' else 2.5
#    if crypto == 'ETH-USD':
#        return 1 if arg_type == 'R' else 2.5
#    if crypto == 'SOL-USD':
#        return 1 if arg_type == 'R' else 2.5
#    if crypto == 'BNB-USD':
#        return 1 if arg_type == 'R' else 2.5
#    if crypto == 'ADA-USD':
#        return 1 if arg_type == 'R' else 2.5
#    if crypto == 'LINK-EUR':
#        return 1 if arg_type == 'R' else 2.5
#    if crypto == 'AVAX-USD':
#        return 1 if arg_type == 'R' else 2.5
#    if crypto == 'DOGE-USD':
#        return 1 if arg_type == 'R' else 2.5
#    if crypto == 'DOT-USD':
#        return 1 if arg_type == 'R' else 2.5
#    if crypto == 'TRX-EUR':
#        return 1 if arg_type == 'R' else 2.5
#    if crypto == 'XRP-USD':
#        return 1 if arg_type == 'R' else 2.5
#    if crypto == 'LTC-USD':
#        return 1 if arg_type == 'R' else 2.5
#    if crypto == 'BCH-USD':
#        return 1 if arg_type == 'R' else 2.5
#    if crypto == 'NEAR-USD':
#        return 1 if arg_type == 'R' else 2.5
#    if crypto == 'UNI7083-USD':
#        return 1 if arg_type == 'R' else 2.5


def GetArg(arg_type, granularity):
    if arg_type == 'lifespan':
        return 20

    if arg_type == 'atr':
        if granularity == '1d':
            return 14
        elif granularity == '30m':
            return 48

    elif arg_type == 'blg':
        if granularity == '1d':
            return [20, 2]
        elif granularity == '30m':
            return [48, 2]

    elif arg_type == 'ema':
        if granularity == '1d':
            return 9
        elif granularity == '30m':
            return 36

    elif arg_type == 'rsi':
        if granularity == '1d':
            return 14
        elif granularity == '30m':
            return 28

    elif arg_type == 'sto':
        if granularity == '1d':
            return [14, 3]
        elif granularity == '30m':
            return [42, 3]

    elif arg_type == 'sma':
        if granularity == '1d':
            return 20
        elif granularity == '30m':
            return 40

    elif arg_type == 'wma':
        if granularity == '1d':
            return 10
        elif granularity == '30m':
            return 20

    elif arg_type == 'dmi':
        if granularity == '1d':
            return 14
        elif granularity == '30m':
            return 28

    elif arg_type == 'macd':
        if granularity == '1d':
            return [12, 26, 9]
        elif granularity == '30m':
            return [36, 78, 26]

    elif arg_type == 'cci':
        if granularity == '1d':
            return 20
        elif granularity == '30m':
            return 40

    elif arg_type == 'ppo':
        if granularity == '1d':
            return [12, 26, 9]
        elif granularity == '30m':
            return [36, 78, 26]

    elif arg_type == 'slope':
        if granularity == '1d':
            return 14
        elif granularity == '30m':
            return 28

    elif arg_type == 'z_score':
        if granularity == '1d':
            return 14
        elif granularity == '30m':
            return 28

    elif arg_type == 'cmf':
        if granularity == '1d':
            return 20
        elif granularity == '30m':
            return 14
        



def GetRP(crypto, arg_type):
    if crypto == 'BTC-USD':
        return 0.6 if arg_type == 'R' else 1.5
    if crypto == 'ETH-USD':
        return 0.6 if arg_type == 'R' else 1.5
    if crypto == 'SOL-USD':
        return 0.6 if arg_type == 'R' else 0.9
    if crypto == 'BNB-USD':
        return 0.6 if arg_type == 'R' else 1.2
    if crypto == 'ADA-USD':
        return 0.6 if arg_type == 'R' else 0.9
    if crypto == 'LINK-EUR':
        return 0.6 if arg_type == 'R' else 1.2
    if crypto == 'AVAX-USD':
        return 0.6 if arg_type == 'R' else 1.2
    if crypto == 'DOGE-USD':
        return 0.6 if arg_type == 'R' else 0.9
    if crypto == 'DOT-USD':
        return 0.6 if arg_type == 'R' else 1.2
    if crypto == 'TRX-EUR':
        return 0.6 if arg_type == 'R' else 1.2
    if crypto == 'XRP-USD':
        return 0.6 if arg_type == 'R' else 0.9
    if crypto == 'LTC-USD':
        return 0.6 if arg_type == 'R' else 1.5
    if crypto == 'BCH-USD':
        return 0.6 if arg_type == 'R' else 1.2
    if crypto == 'NEAR-USD':
        return 0.6 if arg_type == 'R' else 1.2
    if crypto == 'UNI7083-USD':
        return 0.6 if arg_type == 'R' else 1.2



def UpdateArgs(args, crypto):
    args.risk = GetRP(crypto, 'R')
    args.profit = GetRP(crypto, 'P')
    return args