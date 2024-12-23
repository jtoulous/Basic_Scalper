import pandas as pd

def ReadDf(csv_file):
    df = pd.read_csv(csv_file)    
    df.rename(columns={
        'timestamp': 'DATETIME',
        'open': 'OPEN',
        'high': 'HIGH',
        'low': 'LOW',
        'close': 'CLOSE',
        'volume': 'VOLUME'
    }, inplace=True)

    columns_to_keep = ['DATETIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
    df = df[columns_to_keep]
    
    if not pd.api.types.is_datetime64_any_dtype(df['DATETIME']):
        df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    return df